const GEMINI_GENERATE_URL =
  "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent";

const GEMINI_EMBED_URL =
  "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent";

const EMBEDDING_DIMS = 768;

async function generateEmbedding(text) {
  const input = String(text || "").trim();
  if (!input) return null;

  const key = process.env.GEMINI_API_KEY;
  const res = await fetch(`${GEMINI_EMBED_URL}?key=${key}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      content: { parts: [{ text: input }], role: "user" },
      taskType: "RETRIEVAL_DOCUMENT",
      outputDimensionality: EMBEDDING_DIMS,
    }),
  });

  if (!res.ok) throw new Error(`Gemini embed API ${res.status}`);
  const data = await res.json();
  const values = data.embedding?.values;

  if (!Array.isArray(values) || values.length === 0) {
    throw new Error("Empty embedding returned.");
  }

  return values;
}

async function searchKnowledge(supabase, embedding) {
  if (!Array.isArray(embedding) || embedding.length === 0) return [];

  const { data, error } = await supabase.rpc("match_knowledge", {
    query_embedding: embedding,
    match_count: 3,
  });

  if (error) {
    if (error.code !== "PGRST202") {
      console.warn("Knowledge search error:", error.message);
    }
    return [];
  }

  return (data || []).map((r) => r.content).filter(Boolean);
}

async function searchKnowledgeChunks(supabase, embedding, limit = 5) {
  if (!Array.isArray(embedding) || embedding.length === 0) return [];

  const { data, error } = await supabase.rpc("match_knowledge", {
    query_embedding: embedding,
    match_count: limit,
  });

  if (error) {
    if (error.code !== "PGRST202") {
      console.warn("Knowledge search error:", error.message);
    }
    return [];
  }

  return (data || [])
    .map((row) => ({
      id: row.id || null,
      content: row.content || "",
      source: row.source || null,
      similarity: typeof row.similarity === "number" ? row.similarity : null,
    }))
    .filter((row) => row.content);
}

function isBillingQuestion(userInput) {
  const text = String(userInput || "").toLowerCase();
  if (!text) return false;
  return /(bill|billing|amount|due|due date|payment|paid|invoice|how much)/.test(text);
}

function buildExactBillingReply(userInput, latestBill) {
  if (!latestBill || !isBillingQuestion(userInput)) return null;

  const askAmount = /(amount|how much|bill)/i.test(userInput || "");
  const askDue = /(due|due date|last date|deadline)/i.test(userInput || "");
  const askStatus = /(paid|payment status|status)/i.test(userInput || "");

  const amount = latestBill.amount != null ? String(latestBill.amount) : null;
  const dueDate = latestBill.due_date ? String(latestBill.due_date) : null;
  const status = latestBill.status ? String(latestBill.status) : null;
  const month = latestBill.month ? String(latestBill.month) : null;

  const details = [];
  if (askAmount && amount) details.push(`your bill amount${month ? ` for ${month}` : ""} is ${amount}`);
  if (askDue && dueDate) details.push(`the due date is ${dueDate}`);
  if (askStatus && status) details.push(`the payment status is ${status}`);

  if (details.length === 0) {
    if (amount || dueDate || status) {
      const fallback = [
        amount ? `amount ${amount}` : null,
        dueDate ? `due date ${dueDate}` : null,
        status ? `status ${status}` : null,
      ].filter(Boolean).join(", ");
      return `I checked your latest bill details: ${fallback}.`;
    }
    return null;
  }

  return `I checked your account, ${details.join(", and ")}.`;
}

async function generateSuggestedReply(userInput, contextChunks, tier = "Regular", customerData = null) {
  const exactBillingReply = buildExactBillingReply(userInput, customerData?.latestBill);

  const context = contextChunks.length > 0 ? contextChunks.join("\n\n") : null;

  // Build customer-specific account section
  const accountLines = [];
  if (customerData?.name)           accountLines.push(`Customer name: ${customerData.name}`);
  if (customerData?.tier)           accountLines.push(`Tier: ${customerData.tier}`);
  if (customerData?.billingContext) accountLines.push(`Recent bills:\n${customerData.billingContext}`);
  const accountSection = accountLines.length > 0
    ? `Customer account information:\n${accountLines.join("\n")}`
    : null;

  const prompt = [
    `You are coaching a call center agent. A ${tier} tier customer said:`,
    `"${userInput}"`,
    "",
    accountSection || "",
    context
      ? `Relevant knowledge base context:\n${context}`
      : "No specific knowledge base context found.",
    "",
    "Write the exact words the agent should say in response.",
    "1-2 sentences. Professional, empathetic, and direct.",
    accountSection
      ? "If the customer is asking about their bill or account, refer to the specific bill details above."
      : "",
    "Return only the reply text — no labels, no quotes.",
  ].filter((line) => line !== "").join("\n");

  const key = process.env.GEMINI_API_KEY;
  const res = await fetch(`${GEMINI_GENERATE_URL}?key=${key}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: { maxOutputTokens: 150, temperature: 0.3 },
    }),
  });

  if (!res.ok) throw new Error(`Gemini API ${res.status}`);
  const data = await res.json();
  const suggested = (data.candidates?.[0]?.content?.parts?.[0]?.text || "").trim();

  if (exactBillingReply && suggested) {
    return `${exactBillingReply}\n\nSuggested phrasing: ${suggested}`;
  }
  if (exactBillingReply) {
    return `${exactBillingReply}\n\nSuggested phrasing: I can also help explain the bill breakdown and payment options if needed.`;
  }
  return suggested;
}

module.exports = {
  generateEmbedding,
  searchKnowledge,
  searchKnowledgeChunks,
  generateSuggestedReply,
};
