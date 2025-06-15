const express = require('express');
const router = express.Router();
const multer = require('multer');
const { spawn } = require('child_process');
const upload = multer({ dest: 'temp/' });

router.post('/voice-translate', upload.single('voice_file'), async (req, res) => {
  const { target_lang } = req.body;
  const filePath = req.file?.path;

  if (!filePath || !target_lang) {
    return res.status(400).json({ error: 'Missing audio file or target_lang' });
  }

  try {
    const whisper = spawn('python3', ['modules/py/whisper_emotion_translation.py', filePath, target_lang]);
    let output = '';

    whisper.stdout.on('data', (data) => { output += data.toString(); });
    whisper.stderr.on('data', (err) => { console.error('[WhisperError]', err.toString()); });
    whisper.on('close', () => {
      try {
        const result = JSON.parse(output);
        res.json(result);
      } catch (err) {
        res.status(500).json({ error: 'Failed to parse Whisper output', raw: output });
      }
    });
  } catch (err) {
    res.status(500).json({ error: 'Voice translation error', detail: err });
  }
});

module.exports = router;