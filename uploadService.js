const multer = require("multer");
const fs     = require("fs");
const os     = require("os");

// pdf-parse and mammoth are lazy-loaded inside extractText so they don't
// bloat the heap at startup — only loaded when a file is actually uploaded.

// Write to /tmp instead of RAM — avoids heap pressure on Render's free tier
const upload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, os.tmpdir()),
    filename:    (_req, file, cb) => {
      const unique = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
      cb(null, `${unique}-${file.originalname}`);
    },
  }),
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB max
  fileFilter: (_req, file, cb) => {
    const allowed = [
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "application/msword",
      "text/plain",
      "text/csv",
    ];
    const ext = "." + file.originalname.split(".").pop().toLowerCase();
    const extAllowed = [".pdf", ".docx", ".doc", ".txt", ".csv"];

    if (allowed.includes(file.mimetype) || extAllowed.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error("Unsupported file type. Upload PDF, Word (.docx), or text files."));
    }
  },
});

async function extractText(file) {
  const filePath = file.path;
  const ext      = file.originalname.split(".").pop().toLowerCase();

  try {
    if (ext === "pdf") {
      const pdf    = require("pdf-parse");
      const buffer = fs.readFileSync(filePath);
      const data   = await pdf(buffer);
      return data.text || "";
    }

    if (ext === "docx" || ext === "doc") {
      const mammoth = require("mammoth");
      const result  = await mammoth.extractRawText({ path: filePath });
      return result.value || "";
    }

    // txt / csv — read as text
    return fs.readFileSync(filePath, "utf-8");
  } finally {
    // Always delete the temp file to keep /tmp clean
    try { fs.unlinkSync(filePath); } catch { /* ignore */ }
  }
}

/**
 * Split text into chunks of ~1500 characters with 100-char overlap.
 */
function chunkText(text, chunkSize = 1500, overlap = 100) {
  const normalized = text.replace(/\r\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
  if (!normalized) return [];

  const chunks = [];
  let start = 0;

  while (start < normalized.length) {
    let end = Math.min(start + chunkSize, normalized.length);

    if (end < normalized.length) {
      const paraBreak = normalized.lastIndexOf("\n\n", end);
      const sentBreak = normalized.lastIndexOf(". ", end);

      if (paraBreak > start + chunkSize / 2) {
        end = paraBreak + 2;
      } else if (sentBreak > start + chunkSize / 2) {
        end = sentBreak + 2;
      }
    }

    const chunk = normalized.slice(start, end).trim();
    if (chunk) chunks.push(chunk);

    start = end - overlap;
    if (start >= normalized.length) break;
  }

  return chunks;
}

module.exports = { upload, extractText, chunkText };
