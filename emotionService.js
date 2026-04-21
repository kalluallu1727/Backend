const EMOTIONS = ["calm", "confused", "frustrated", "angry"];

function normalizeEmotion(raw) {
  const cleaned = String(raw || "").trim().toLowerCase();
  return EMOTIONS.includes(cleaned) ? cleaned : "calm";
}

module.exports = { EMOTIONS, normalizeEmotion };
