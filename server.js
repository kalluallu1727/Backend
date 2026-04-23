// =============================================================
// Required env vars (add to .env):
//   SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, GEMINI_API_KEY
//   TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
//   TWILIO_API_KEY, TWILIO_API_SECRET
//   TWILIO_TWIML_APP_SID
//   TWILIO_PHONE_NUMBER
//   APP_BASE_URL
//   CORS_ORIGIN, PORT
// =============================================================

const express  = require("express");
const http     = require("http");
const { randomUUID } = require("crypto");
const cors     = require("cors");
const { createClient } = require("@supabase/supabase-js");
const twilio   = require("twilio");
const { analyzeCustomerSpeech }                            = require("./decisionEngine");
const { generateEmbedding, searchKnowledge, generateSuggestedReply } = require("./ragService");
const { upload, extractText, chunkText }                   = require("./uploadService");
require("dotenv").config({ quiet: true });

// ── App + HTTP server ─────────────────────────────────────────
const app    = express();
const server = http.createServer(app);

// ── CORS ──────────────────────────────────────────────────────
const corsOrigin = process.env.CORS_ORIGIN
  ? process.env.CORS_ORIGIN.split(",").map((o) => o.trim()).filter(Boolean)
  : true;

app.use(cors({ origin: corsOrigin }));
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// ── Env validation ────────────────────────────────────────────
const {
  SUPABASE_URL,
  SUPABASE_SERVICE_ROLE_KEY,
  GEMINI_API_KEY,
  APP_BASE_URL,
  TWILIO_ACCOUNT_SID,
  TWILIO_AUTH_TOKEN,
  TWILIO_API_KEY,
  TWILIO_API_SECRET,
  TWILIO_TWIML_APP_SID,
  TWILIO_PHONE_NUMBER,
  PORT,
} = process.env;

const MISSING_VARS = [
  ["SUPABASE_URL",            SUPABASE_URL],
  ["SUPABASE_SERVICE_ROLE_KEY", SUPABASE_SERVICE_ROLE_KEY],
  ["GEMINI_API_KEY",          GEMINI_API_KEY],
].filter(([, v]) => !v).map(([k]) => k);

if (MISSING_VARS.length > 0) {
  console.error(`[startup] Missing required environment variable(s): ${MISSING_VARS.join(", ")}`);
  console.error("[startup] Set these in your Render dashboard → Environment, then redeploy.");
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

// Twilio REST client (optional — only needed for dialling agents)
const twilioClient =
  TWILIO_ACCOUNT_SID && TWILIO_AUTH_TOKEN
    ? twilio(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    : null;

const BASE_URL = (APP_BASE_URL || "").replace(/\/+$/, "");
const AGENT_IDENTITIES = {
  "1": process.env.AGENT_IDENTITY_BILLING || "agent_billing",
  "2": process.env.AGENT_IDENTITY_SUPPORT || "agent_support",
  "3": process.env.AGENT_IDENTITY_GENERAL || "agent",
};
const FALLBACK_AGENT_IDENTITY = process.env.AGENT_IDENTITY_FALLBACK || "agent";
const ENABLE_SKILL_ROUTING = String(process.env.ENABLE_SKILL_ROUTING || "false").toLowerCase() === "true";
const activeAgentCallByIdentity = new Map();
const callAgentIdentityByCallId = new Map();

const port          = Number(PORT) || 3000;
const hasExplicitPort = Boolean(PORT);

// ── Startup diagnostics ──────────────────────────────────────
console.log("[startup] APP_BASE_URL    :", BASE_URL || "(not set!)");
console.log("[startup] Twilio client   :", twilioClient ? "configured" : "NOT configured");

// ── Helpers ───────────────────────────────────────────────────
function escapeXml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function saveIvrMessage(callId, content) {
  const text = String(content || "").trim();
  if (!callId || !text) return;
  supabase
    .from("messages")
    .insert({ call_id: callId, role: "ivr", content: text })
    .then(({ error }) => {
      if (error) console.error("[ivr] Message insert error:", error.message);
    });
}

function buildIvrMenuTwiml(callId, isCustomerKnown, retry = false) {
  const actionUrl = escapeXml(`${BASE_URL}/api/twilio/ivr/select?call_id=${callId}`);
  const retryUrl = escapeXml(`${BASE_URL}/api/twilio/ivr/select?call_id=${callId}&retry=1`);
  const retryLine = retry ? `<Say voice="alice">Sorry, I did not get a valid option.</Say>` : "";
  const verificationLine = isCustomerKnown
    ? `<Say voice="alice">Customer verification complete from your registered number.</Say>`
    : `<Say voice="alice">Customer verification step complete.</Say>`;

  return `<?xml version="1.0" encoding="UTF-8"?>
<Response>
  ${verificationLine}
  ${retryLine}
  <Gather input="dtmf" numDigits="1" timeout="6" action="${actionUrl}" method="POST">
    <Say voice="alice">Press 1 for billing.</Say>
    <Say voice="alice">Press 2 for service related questions.</Say>
    <Say voice="alice">Press 3 for General Queries.</Say>
  </Gather>
  <Redirect method="POST">${retryUrl}</Redirect>
</Response>`;
}

function buildProblemCaptureTwiml(callId, optionDigit, retryCount = 0) {
  const actionUrl = escapeXml(`${BASE_URL}/api/twilio/ivr/problem?call_id=${callId}&ivr_option=${optionDigit}&retry_count=${retryCount}`);
  const timeoutUrl = escapeXml(`${BASE_URL}/api/twilio/ivr/problem?call_id=${callId}&ivr_option=${optionDigit}&timeout=1&retry_count=${retryCount}`);
  const retryPrompt = retryCount > 0
    ? `<Say voice="alice">I have not received any input. Please tell your query or problem in a few words.</Say>`
    : "";
  return `<?xml version="1.0" encoding="UTF-8"?>
<Response>
  ${retryPrompt}
  <Gather input="speech" speechTimeout="auto" timeout="7" action="${actionUrl}" method="POST">
    <Say voice="alice">In a few words, please tell me your problem or query.</Say>
  </Gather>
  <Redirect method="POST">${timeoutUrl}</Redirect>
</Response>`;
}

function ivrLabelFromDigit(digit) {
  if (digit === "1") return "Billing";
  if (digit === "2") return "Service Related Questions";
  if (digit === "3") return "General Queries";
  return "General";
}

function getAgentIdentityFromOption(optionDigit) {
  if (!ENABLE_SKILL_ROUTING) return FALLBACK_AGENT_IDENTITY;
  return AGENT_IDENTITIES[String(optionDigit || "").trim()] || AGENT_IDENTITIES["3"];
}

function isAgentIdentityBusy(agentIdentity, callId) {
  const activeCallId = activeAgentCallByIdentity.get(agentIdentity);
  return Boolean(activeCallId && activeCallId !== callId);
}

function dialAgentForCall(callId, optionDigit) {
  const agentIdentity = getAgentIdentityFromOption(optionDigit);
  if (isAgentIdentityBusy(agentIdentity, callId)) {
    const activeCallId = activeAgentCallByIdentity.get(agentIdentity);
    console.log(`[voice] Agent ${agentIdentity} busy on call ${activeCallId}; cannot dial ${callId} yet.`);
    return false;
  }
  if (!twilioClient || !TWILIO_PHONE_NUMBER) {
    console.warn("[voice] ERROR: Twilio client not configured — agent NOT dialled.");
    console.warn(`[voice]   TWILIO_ACCOUNT_SID : ${TWILIO_ACCOUNT_SID ? "set" : "MISSING"}`);
    console.warn(`[voice]   TWILIO_AUTH_TOKEN  : ${TWILIO_AUTH_TOKEN  ? "set" : "MISSING"}`);
    console.warn(`[voice]   TWILIO_PHONE_NUMBER: ${TWILIO_PHONE_NUMBER ? "set" : "MISSING"}`);
    return false;
  }
  const agentUrl = `${BASE_URL}/api/twilio/agent?call_id=${callId}&agent_identity=${encodeURIComponent(agentIdentity)}`;
  const startDial = (identity) => {
    activeAgentCallByIdentity.set(identity, callId);
    callAgentIdentityByCallId.set(callId, identity);
    saveIvrMessage(callId, `IVR_AGENT:client:${identity}`);
    return twilioClient.calls.create({
      to: `client:${identity}`,
      from: TWILIO_PHONE_NUMBER,
      url: `${BASE_URL}/api/twilio/agent?call_id=${callId}&agent_identity=${encodeURIComponent(identity)}`,
    });
  };

  console.log(`[voice] Dialling agent → client:${agentIdentity}`);
  console.log(`[voice] Agent TwiML URL : ${agentUrl}`);
  startDial(agentIdentity).then((call) => {
    console.log(`[voice] Agent call created ✓ SID=${call.sid}`);
  }).catch((err) => {
    if (activeAgentCallByIdentity.get(agentIdentity) === callId) {
      activeAgentCallByIdentity.delete(agentIdentity);
    }
    callAgentIdentityByCallId.delete(callId);
    console.error(`[voice] Agent dial FAILED (${agentIdentity}): ${err.message}`);
    console.error(`[voice] Twilio error code : ${err.code || "n/a"}`);
    console.error(`[voice] Twilio more info  : ${err.moreInfo || "n/a"}`);

    if (FALLBACK_AGENT_IDENTITY === agentIdentity || isAgentIdentityBusy(FALLBACK_AGENT_IDENTITY, callId)) return;

    console.log(`[voice] Attempting fallback agent identity client:${FALLBACK_AGENT_IDENTITY}`);
    startDial(FALLBACK_AGENT_IDENTITY).then((fallbackCall) => {
      console.log(`[voice] Fallback agent call created ✓ SID=${fallbackCall.sid}`);
    }).catch((fallbackErr) => {
      if (activeAgentCallByIdentity.get(FALLBACK_AGENT_IDENTITY) === callId) {
        activeAgentCallByIdentity.delete(FALLBACK_AGENT_IDENTITY);
      }
      callAgentIdentityByCallId.delete(callId);
      console.error(`[voice] Fallback agent dial FAILED (${FALLBACK_AGENT_IDENTITY}): ${fallbackErr.message}`);
    });
  });
  return true;
}

function buildAgentWaitingTwiml(callId, waitCount = 0, optionDigit = "3") {
  const nextCount = Number(waitCount || 0) + 1;
  const loopUrl = escapeXml(`${BASE_URL}/api/twilio/wait-loop?call_id=${callId}&wait_count=${nextCount}&ivr_option=${optionDigit}`);
  const intro = waitCount === 0
    ? `<Say voice="alice">All our agents are currently helping other customers. Please stay on the line while we connect you.</Say>`
    : `<Say voice="alice">Thank you for waiting. We will connect you to the next available agent shortly.</Say>`;
  return `<?xml version="1.0" encoding="UTF-8"?>
<Response>
  ${intro}
  <Play>https://com.twilio.music.classical.s3.amazonaws.com/BusyStrings.mp3</Play>
  <Pause length="3" />
  <Redirect method="POST">${loopUrl}</Redirect>
</Response>`;
}

function buildPhoneVariants(rawPhone) {
  const digitsOnly = String(rawPhone || "").replace(/\D/g, "");
  if (!digitsOnly) return [rawPhone].filter(Boolean);

  const lastTen = digitsOnly.slice(-10);
  const variants = new Set([rawPhone, digitsOnly, `+${digitsOnly}`]);

  if (digitsOnly.length === 10) {
    variants.add(`1${digitsOnly}`);
    variants.add(`+1${digitsOnly}`);
  }
  if (digitsOnly.length > 10) {
    variants.add(lastTen);
    variants.add(`+1${lastTen}`);
    variants.add(`1${lastTen}`);
  }

  return Array.from(variants).filter(Boolean);
}

async function lookupCustomerByPhone(rawPhone) {
  if (!rawPhone) return null;

  const variants = buildPhoneVariants(rawPhone);
  if (variants.length === 0) return null;

  const { data, error } = await supabase
    .from("customers")
    .select("*")
    .in("phone", variants)
    .limit(1)
    .maybeSingle();

  if (error) {
    console.warn("Customer lookup failed:", error.message);
    return null;
  }
  return data || null;
}

// ── Conference TwiML builders ─────────────────────────────────
// Twilio's <Start><Transcription> handles speech-to-text natively.
// Final transcripts are POSTed to /api/transcription by Twilio.
function customerConferenceTwiml(callId) {
  const transcriptionUrl = escapeXml(`${BASE_URL}/api/transcription?call_id=${callId}&role=customer`);
  const statusUrl        = escapeXml(`${BASE_URL}/api/conference-status?call_id=${callId}`);
  const room             = `room-${callId}`;
  console.log(`[twiml] Transcription callback: ${BASE_URL}/api/transcription?call_id=${callId}&role=customer`);

  return `<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">Please hold while we connect you to an agent.</Say>
  <Start>
    <Transcription statusCallbackUrl="${transcriptionUrl}"
                   statusCallbackMethod="POST"
                   track="inbound_track" />
  </Start>
  <Dial>
    <Conference beep="false"
                waitUrl="https://twimlets.com/holdmusic?Bucket=com.twilio.music.classical"
                endConferenceOnExit="true"
                statusCallbackEvent="end"
                statusCallback="${statusUrl}">
      ${room}
    </Conference>
  </Dial>
</Response>`;
}

function agentConferenceTwiml(callId) {
  const transcriptionUrl = escapeXml(`${BASE_URL}/api/transcription?call_id=${callId}&role=agent`);
  const room             = `room-${callId}`;

  return `<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Start>
    <Transcription statusCallbackUrl="${transcriptionUrl}"
                   statusCallbackMethod="POST"
                   track="inbound_track" />
  </Start>
  <Dial>
    <Conference beep="false" waitUrl="" endConferenceOnExit="true">
      ${room}
    </Conference>
  </Dial>
</Response>`;
}

// ── Routes ────────────────────────────────────────────────────

app.get("/", (_req, res) => res.send("Agent-assist IVR server running."));

// -- Twilio Access Token for agent browser softphone ----------
app.get("/api/token", (_req, res) => {
  if (!TWILIO_ACCOUNT_SID || !TWILIO_API_KEY || !TWILIO_API_SECRET) {
    return res.status(503).json({ error: "Twilio credentials not configured." });
  }

  const AccessToken = twilio.jwt.AccessToken;
  const VoiceGrant  = AccessToken.VoiceGrant;

  const token = new AccessToken(
    TWILIO_ACCOUNT_SID,
    TWILIO_API_KEY,
    TWILIO_API_SECRET,
    { identity: "agent", ttl: 3600 }
  );

  const voiceGrant = new VoiceGrant({
    outgoingApplicationSid: TWILIO_TWIML_APP_SID || undefined,
    incomingAllow: true,
  });

  token.addGrant(voiceGrant);
  return res.json({ token: token.toJwt() });
});

// -- Customer calls in → Conference TwiML + dial agent --------
app.post("/api/twilio/voice", async (req, res) => {
  res.set("Content-Type", "text/xml");

  const callId      = randomUUID();
  const callerPhone = String(req.body.From || req.body.Caller || "").trim() || null;

  console.log(`\n[voice] ── Incoming call ──────────────────────`);
  console.log(`[voice] callId       : ${callId}`);
  console.log(`[voice] callerPhone  : ${callerPhone || "(unknown)"}`);
  console.log(`[voice] BASE_URL     : ${BASE_URL    || "(NOT SET)"}`);
  console.log(`[voice] twilioClient : ${twilioClient ? "OK" : "MISSING"}`);
  console.log(`[voice] TWILIO_PHONE : ${TWILIO_PHONE_NUMBER || "(NOT SET)"}`);

  try {
    const customer = callerPhone ? await lookupCustomerByPhone(callerPhone) : null;
    console.log(`[voice] customer     : ${customer ? `${customer.name} / tier=${customer.tier}` : "not found in DB"}`);

    // Insert call record immediately (fire-and-forget)
    supabase.from("calls").insert({
      id:             callId,
      customer_phone: customer?.phone || callerPhone,
      customer_name:  customer?.name  || null,
      tier:           customer?.tier  || null,
      priority:       "low",
    }).then(({ error }) => {
      if (error) console.error("[voice] Call insert error:", error.message);
      else console.log(`[voice] Call inserted in DB ✓`);
    });

    saveIvrMessage(callId, customer ? "IVR_VERIFICATION:Verified customer from registered number." : "IVR_VERIFICATION:Verification step completed.");
    res.send(buildIvrMenuTwiml(callId, Boolean(customer)));
    console.log(`[voice] IVR menu TwiML sent ✓`);
  } catch (error) {
    console.error("[voice] Unexpected error:", error.message);
    res.send(buildIvrMenuTwiml(callId, false));
  }
});

app.post("/api/twilio/wait-loop", (req, res) => {
  res.set("Content-Type", "text/xml");
  const callId = String(req.query.call_id || "").trim();
  const waitCount = Number(req.query.wait_count || 0);
  const optionDigit = String(req.query.ivr_option || "3").trim();
  if (!callId) return res.send("<Response><Hangup/></Response>");

  const targetIdentity = getAgentIdentityFromOption(optionDigit);
  if (isAgentIdentityBusy(targetIdentity, callId)) {
    return res.send(buildAgentWaitingTwiml(callId, waitCount, optionDigit));
  }

  const dialStarted = dialAgentForCall(callId, optionDigit);
  if (!dialStarted) {
    return res.send(buildAgentWaitingTwiml(callId, waitCount, optionDigit));
  }
  saveIvrMessage(callId, "IVR_WAITING:Agent became available. Connecting now.");
  return res.send(customerConferenceTwiml(callId));
});

app.post("/api/twilio/ivr/select", (req, res) => {
  res.set("Content-Type", "text/xml");
  const callId = String(req.query.call_id || "").trim();
  const digit = String(req.body.Digits || "").trim();
  if (!callId) return res.send("<Response><Hangup/></Response>");

  const optionLabel = ivrLabelFromDigit(digit);
  if (!["1", "2", "3"].includes(digit)) {
    return res.send(buildIvrMenuTwiml(callId, false, true));
  }

  saveIvrMessage(callId, `IVR_SELECTION:#${digit} ${optionLabel}`);
  res.send(buildProblemCaptureTwiml(callId, digit));
});

app.post("/api/twilio/ivr/problem", (req, res) => {
  res.set("Content-Type", "text/xml");
  const callId = String(req.query.call_id || "").trim();
  const optionDigit = String(req.query.ivr_option || "").trim();
  const retryCount = Number(req.query.retry_count || 0);
  const spoken = String(req.body.SpeechResult || "").replace(/\s+/g, " ").trim();
  if (!callId) return res.send("<Response><Hangup/></Response>");

  if (!spoken) {
    const nextRetry = retryCount + 1;
    if (nextRetry <= 1) {
      saveIvrMessage(callId, "IVR_EVENT:No input received. Re-prompting once.");
      return res.send(buildProblemCaptureTwiml(callId, optionDigit, nextRetry));
    }
    saveIvrMessage(callId, "IVR_EVENT:No customer query captured after retry. Call disconnected.");
    return res.send(`<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">We did not receive your query. Please call again. Goodbye.</Say>
  <Hangup />
</Response>`);
  }

  const summary = spoken || "No customer query captured.";
  saveIvrMessage(callId, `IVR_SUMMARY:${summary}`);
  saveIvrMessage(callId, `IVR_SELECTION:#${optionDigit || "-"} ${ivrLabelFromDigit(optionDigit)}`);

  // Connect immediately when no active call is using the agent.
  // Only play waiting flow when the agent is actually busy.
  const targetIdentity = getAgentIdentityFromOption(optionDigit);
  if (!isAgentIdentityBusy(targetIdentity, callId)) {
    const dialStarted = dialAgentForCall(callId, optionDigit);
    if (dialStarted) {
      saveIvrMessage(callId, `IVR_WAITING:Agent ${targetIdentity} available. Connecting now.`);
      return res.send(customerConferenceTwiml(callId));
    }
  }

  saveIvrMessage(callId, `IVR_WAITING:Agent ${targetIdentity} busy. Playing waiting message before connection.`);
  return res.send(buildAgentWaitingTwiml(callId, 0, optionDigit));
});

// -- Agent leg TwiML (called by Twilio when agent answers) ----
app.post("/api/twilio/agent", (req, res) => {
  res.set("Content-Type", "text/xml");

  const callId = String(req.query.call_id || "").trim();
  if (!callId) {
    return res.send("<Response><Hangup/></Response>");
  }

  res.send(agentConferenceTwiml(callId));
});

// -- Outbound call: agent leg (TwiML App calls this when agent dials) -
app.post("/api/twilio/outbound", async (req, res) => {
  res.set("Content-Type", "text/xml");
  const to = String(req.body.To || "").trim();

  if (!to || !TWILIO_PHONE_NUMBER || !twilioClient) {
    return res.send("<Response><Hangup/></Response>");
  }

  const callId = randomUUID();
  const room   = `room-${callId}`;

  console.log(`\n[outbound] ── Outbound call ───────────────────`);
  console.log(`[outbound] callId : ${callId}`);
  console.log(`[outbound] to     : ${to}`);

  // Insert call record then update with customer info if found
  supabase.from("calls").insert({
    id:             callId,
    customer_phone: to,
    customer_name:  null,
    tier:           null,
    priority:       "low",
  }).then(({ error }) => {
    if (error) console.error("[outbound] Call insert error:", error.message);
    else console.log(`[outbound] Call inserted in DB ✓`);
  });

  lookupCustomerByPhone(to).then((customer) => {
    if (!customer) return;
    supabase.from("calls").update({ customer_name: customer.name, tier: customer.tier })
      .eq("id", callId).then(({ error }) => {
        if (error) console.error("[outbound] Customer update error:", error.message);
      });
  });

  // Dial the customer into the same conference room via REST API
  const customerUrl = `${BASE_URL}/api/twilio/outbound-customer?call_id=${callId}`;
  twilioClient.calls.create({ to, from: TWILIO_PHONE_NUMBER, url: customerUrl })
    .then((call) => console.log(`[outbound] Customer dialed ✓ SID=${call.sid}`))
    .catch((err) => console.error(`[outbound] Customer dial FAILED: ${err.message}`));

  // Put agent into the conference room with transcription
  const agentTranscriptionUrl = escapeXml(`${BASE_URL}/api/transcription?call_id=${callId}&role=agent`);
  const statusUrl              = escapeXml(`${BASE_URL}/api/conference-status?call_id=${callId}`);

  res.send(`<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Start>
    <Transcription statusCallbackUrl="${agentTranscriptionUrl}"
                   statusCallbackMethod="POST"
                   track="inbound_track" />
  </Start>
  <Dial>
    <Conference beep="false"
                waitUrl="https://twimlets.com/holdmusic?Bucket=com.twilio.music.classical"
                statusCallbackEvent="end"
                statusCallback="${statusUrl}">
      ${room}
    </Conference>
  </Dial>
</Response>`);
  console.log(`[outbound] Agent TwiML sent ✓`);
});

// -- Outbound call: customer leg (Twilio calls this when customer answers) -
app.post("/api/twilio/outbound-customer", (req, res) => {
  res.set("Content-Type", "text/xml");
  const callId = String(req.query.call_id || "").trim();
  if (!callId) return res.send("<Response><Hangup/></Response>");

  const room                    = `room-${callId}`;
  const customerTranscriptionUrl = escapeXml(`${BASE_URL}/api/transcription?call_id=${callId}&role=customer`);

  res.send(`<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Start>
    <Transcription statusCallbackUrl="${customerTranscriptionUrl}"
                   statusCallbackMethod="POST"
                   track="inbound_track" />
  </Start>
  <Dial>
    <Conference beep="false" waitUrl="">
      ${room}
    </Conference>
  </Dial>
</Response>`);
});

// -- Conference status callback (called when conference ends) -
app.post("/api/conference-status", (req, res) => {
  const callId = String(req.query.call_id || "").trim();
  const event  = req.body.StatusCallbackEvent;

  if (event === "conference-end" && callId) {
    const mappedIdentity = callAgentIdentityByCallId.get(callId);
    if (mappedIdentity && activeAgentCallByIdentity.get(mappedIdentity) === callId) {
      activeAgentCallByIdentity.delete(mappedIdentity);
    }
    callAgentIdentityByCallId.delete(callId);
    console.log(`Conference ended callId=${callId}`);
  }

  res.status(200).end();
});

// -- Fetch customer's recent bills from DB (graceful — works even if table missing) --
async function fetchCustomerBillingContext(customerId, customerPhone, customerName) {
  // Try to query a `bills` table; adapt to whatever columns exist.
  try {
    let query = supabase.from("bills").select("*").order("created_at", { ascending: false }).limit(3);
    if (customerId)     query = query.eq("customer_id", customerId);
    else if (customerPhone) query = query.or(`customer_phone.eq.${customerPhone},phone.eq.${customerPhone}`);
    else if (customerName)  query = query.ilike("customer_name", `%${customerName}%`);
    else return null;

    const { data, error } = await query;
    if (error || !data || data.length === 0) return null;

    return data.map((row) => {
      const parts = [];
      const month  = row.bill_month  || row.month         || row.billing_period || row.period || null;
      const amount = row.amount      || row.total_amount  || row.bill_amount    || null;
      const due    = row.due_date    || row.due           || null;
      const status = row.status      || row.payment_status|| null;
      const paid   = row.paid_date   || row.payment_date  || null;
      if (month)  parts.push(`Month: ${month}`);
      if (amount) parts.push(`Amount: ${amount}`);
      if (due)    parts.push(`Due: ${due}`);
      if (status) parts.push(`Status: ${status}`);
      if (paid)   parts.push(`Paid on: ${paid}`);
      // Fall back: any remaining string columns
      if (parts.length === 0) {
        Object.entries(row).forEach(([k, v]) => {
          if (k !== "id" && k !== "customer_id" && v != null)
            parts.push(`${k}: ${v}`);
        });
      }
      return parts.join(", ");
    }).filter(Boolean).join("\n");
  } catch {
    return null;
  }
}

// -- Twilio transcription webhook -----------------------------
// Twilio calls this for every transcription event.
// We only act on final transcripts (Final=true).
app.post("/api/transcription", async (req, res) => {
  // Respond immediately so Twilio doesn't retry.
  res.status(200).end();

  const callId  = String(req.query.call_id || "").trim();
  const role    = String(req.query.role    || "customer").trim();
  const event   = req.body.TranscriptionEvent;
  const isFinal = req.body.Final === "true";

  // Log every event so we can see what Twilio is sending
  console.log(`[transcript] event=${event} final=${req.body.Final} role=${role} callId=${callId}`);

  if (!callId) { console.warn("[transcript] Missing call_id — skipping"); return; }
  if (event !== "transcription-content") { console.log(`[transcript] Skipping event type: ${event}`); return; }
  if (!isFinal) { console.log("[transcript] Partial transcript — waiting for final"); return; }

  let transcript = "";
  try {
    const data = JSON.parse(req.body.TranscriptionData || "{}");
    transcript = String(data.transcript || "").trim();
  } catch {
    console.warn("[transcript] Failed to parse TranscriptionData:", req.body.TranscriptionData);
    return;
  }

  if (!transcript) {
    console.log(`[transcript] [${role}] callId=${callId} — empty final transcript`);
    return;
  }

  console.log(`[transcript] [${role}] "${transcript.slice(0, 120)}"`);

  // Save the utterance to the messages table
  const dbRole = role === "agent" ? "agent" : "user";
  supabase
    .from("messages")
    .insert({ call_id: callId, role: dbRole, content: transcript })
    .then(({ error }) => {
      if (error) console.error("[transcript] Message insert error:", error.message);
      else console.log(`[transcript] Message saved ✓ role=${dbRole}`);
    });

  // Only run the AI analysis pipeline for customer speech
  if (role !== "customer") return;

  try {
    // Fetch full call + customer data for context
    const { data: callData } = await supabase
      .from("calls").select("*").eq("id", callId).maybeSingle();
    const tier         = callData?.tier         || "Regular";
    const customerName = callData?.customer_name || null;
    const customerPhone= callData?.customer_phone|| null;

    // Enrich the embedding query with customer name so we find their specific bills/docs
    const searchQuery = customerName ? `${customerName} ${transcript}` : transcript;

    const [analysisResult, embedding] = await Promise.all([
      analyzeCustomerSpeech(transcript, tier),
      generateEmbedding(searchQuery).catch(() => null),
    ]);

    console.log(`[analysis] emotion=${analysisResult.emotion} intent=${analysisResult.intent} priority=${analysisResult.priority}`);

    // ── Step 1: save basic analysis immediately so frontend shows it right away ──
    const { data: savedRows, error: insertErr } = await supabase
      .from("analysis")
      .insert({
        call_id:           callId,
        emotion:           analysisResult.emotion,
        intent:            analysisResult.intent,
        priority:          analysisResult.priority,
        suggested_actions: analysisResult.suggested_actions,
        suggested_reply:   null,
      })
      .select("id");

    if (insertErr) {
      console.error("[analysis] Insert error:", insertErr.message);
    } else {
      console.log("[analysis] Basic analysis saved ✓");
    }

    // Update call priority immediately
    supabase.from("calls").update({ priority: analysisResult.priority })
      .eq("id", callId)
      .then(({ error }) => { if (error) console.error("[analysis] Priority update:", error.message); });

    // ── Step 2: generate suggested reply and update the row (non-blocking for step 1) ──
    try {
      const [contextChunks, billingContext] = await Promise.all([
        embedding ? searchKnowledge(supabase, embedding) : Promise.resolve([]),
        fetchCustomerBillingContext(null, customerPhone, customerName),
      ]);

      if (billingContext) console.log(`[analysis] billing context found for ${customerName || customerPhone}`);

      const customerData  = { name: customerName, tier, billingContext };
      const suggestedReply = await generateSuggestedReply(transcript, contextChunks, tier, customerData);

      const analysisId = savedRows?.[0]?.id;
      if (analysisId && suggestedReply) {
        supabase.from("analysis")
          .update({ suggested_reply: suggestedReply })
          .eq("id", analysisId)
          .then(({ error }) => {
            if (error) console.error("[analysis] Reply update error:", error.message);
            else console.log("[analysis] Suggested reply saved ✓");
          });
      }
    } catch (replyErr) {
      console.error("[analysis] Suggested reply failed (basic analysis still saved):", replyErr.message);
    }
  } catch (err) {
    console.error("[analysis] Pipeline error:", err.message);
  }
});

// -- Knowledge base: add text content + embedding -------------
app.post("/api/knowledge", async (req, res) => {
  const content = String(req.body.content || "").trim();
  const source  = String(req.body.source  || "").trim();

  if (!content) {
    return res.status(400).json({ error: "content is required." });
  }

  try {
    const embedding = await generateEmbedding(content);

    const { error } = await supabase.from("knowledge_base").insert({
      content,
      source:    source || null,
      embedding,
    });

    if (error) throw new Error(error.message);

    return res.json({ success: true, chunks: 1 });
  } catch (err) {
    console.error("Knowledge insert error:", err.message);
    return res.status(500).json({ error: "Failed to add to knowledge base." });
  }
});

// -- Knowledge base: upload file (PDF / DOCX / TXT / CSV) -----
app.post("/api/knowledge/upload", upload.single("file"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded." });
  }

  const source = req.file.originalname;

  let chunks;
  try {
    const rawText = await extractText(req.file);
    if (!rawText.trim()) {
      return res.status(422).json({ error: "Could not extract text from file." });
    }

    chunks = chunkText(rawText);
    if (chunks.length === 0) {
      return res.status(422).json({ error: "File appears to be empty." });
    }
  } catch (err) {
    console.error("File parse error:", err.message);
    return res.status(422).json({ error: "Failed to parse file content." });
  }

  // Respond immediately — embedding can take minutes for large files.
  res.json({
    success:      true,
    file:         source,
    total_chunks: chunks.length,
    inserted_chunks: chunks.length,
    status:       "processing",
  });

  // Background: embed and insert each chunk sequentially (avoid rate limits)
  (async () => {
    let inserted = 0;
    for (const chunk of chunks) {
      try {
        const embedding = await generateEmbedding(chunk);
        const { error } = await supabase.from("knowledge_base").insert({
          content:   chunk,
          source,
          embedding,
        });
        if (error) {
          console.error(`Chunk insert error (${source}):`, error.message);
        } else {
          inserted++;
        }
      } catch (chunkErr) {
        console.error(`Embedding error (${source}):`, chunkErr.message);
      }
    }
    console.log(`[upload] ${source}: inserted ${inserted}/${chunks.length} chunks`);
  })();
});

// Multer error handler
app.use((err, _req, res, _next) => {
  if (err?.code === "LIMIT_FILE_SIZE") {
    return res.status(413).json({ error: "File too large. Max 10MB." });
  }
  if (err?.message) {
    return res.status(400).json({ error: err.message });
  }
  res.status(500).json({ error: "Unexpected error." });
});

// -- Customer details -----------------------------------------
async function fetchCustomerDetails(req, res, overrides = {}) {
  const id    = String(overrides.id    ?? req.query.id    ?? "").trim();
  const email = String(overrides.email ?? req.query.email ?? "").trim();
  const phone = String(overrides.phone ?? req.query.phone ?? "").trim();

  if (!id && !email && !phone) {
    return res.status(400).json({ error: "Provide id, email, or phone." });
  }

  try {
    let query = supabase.from("customers").select("*").limit(1);

    if (id)    query = query.eq("id", id);
    else if (email) query = query.eq("email", email);
    else       query = query.in("phone", buildPhoneVariants(phone));

    const { data, error } = await query.maybeSingle();

    if (error) {
      console.error("Customer fetch error:", error.message);
      return res.status(500).json({ error: "Failed to fetch customer." });
    }

    if (!data) return res.status(404).json({ error: "Customer not found." });

    return res.json({ customer: data });
  } catch (err) {
    console.error("customer-details error:", err.message);
    return res.status(500).json({ error: "Unexpected error." });
  }
}

app.get("/api/customer-details", (req, res) => fetchCustomerDetails(req, res));
app.get("/api/customers/:id",    (req, res) => fetchCustomerDetails(req, res, { id: req.params.id }));

// ── Start server ──────────────────────────────────────────────
function startServer(p) {
  server.listen(p, () =>
    console.log(`Agent-assist IVR server running on port ${p}`)
  );

  server.on("error", (error) => {
    if (error.code !== "EADDRINUSE") throw error;
    if (hasExplicitPort) {
      console.error(`Port ${p} is in use. Set a different PORT in .env.`);
      process.exit(1);
    }
    console.warn(`Port ${p} busy. Retrying on ${p + 1}...`);
    startServer(p + 1);
  });
}

if (require.main === module) {
  startServer(port);
}

module.exports = { app, server };
