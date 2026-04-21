const HIGH_STAKES_INTENTS = [
  "cancellation",
  "complaint",
  "billing_issue",
  "payment_issue",
  "refund_request",
];

function normalizeIntent(raw) {
  return (
    String(raw || "general_inquiry")
      .toLowerCase()
      .replace(/\s+/g, "_")
      .replace(/[^a-z_]/g, "") || "general_inquiry"
  );
}

function isHighStakesIntent(intent) {
  return HIGH_STAKES_INTENTS.includes(String(intent || "").toLowerCase());
}

module.exports = { HIGH_STAKES_INTENTS, normalizeIntent, isHighStakesIntent };
