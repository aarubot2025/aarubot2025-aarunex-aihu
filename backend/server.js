
const express = require('express');
const cors = require('cors');
const path = require('path');
const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// 👇 Serve static files from the frontend directory
app.use('/frontend', express.static(path.resolve(__dirname, '../frontend')));

// API routes
const voiceTranslationRouter = require('./modules/voiceTranslationRouter');
app.use('/api', voiceTranslationRouter);

// Root endpoint
app.get('/', (req, res) => {
  res.send('🧠 AaruNex LWL Server is running. Visit /frontend/voice.html to begin.');
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 AaruNex Server listening on port ${PORT}`);
});

