const express = require('express');
const app = express();
const PORT = 6003;
const morgan = require('morgan');
const cors = require('cors');
const {spawn} = require('child_process');
const path = require('path');

app.use(cors(
    {
        origin: '*',
        methods: ['GET', 'POST', 'PUT', 'DELETE'],
        allowedHeaders: ['Content-Type', 'Authorization']
    }
));
app.use(morgan('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(express.static('public'));

app.get('/', (req, res) => {
    res.json({ message: 'Hello World' });
});

const getIntents = (text, model) => {
    return new Promise((resolve, reject) => {
        try {
            let pathToClassifyScript = path.join(__dirname, 'src', 'utils', 'classify_intent.py');
            console.log(pathToClassifyScript);
            console.log(text);
            const python = spawn('python3', [pathToClassifyScript, text, model]);
            let dataString = '';
  
            python.stdout.on('data', (data) => {
                dataString += data.toString();
            });
  
            python.stderr.on('data', (data) => {
                console.error(data.toString());
            });
  
            python.on('close', (code) => {
                console.log(`Child process exited with code ${code}`);
                console.log(`Data received from Python script: ${dataString}`);
                try {
                    const intents = JSON.parse(dataString);
                    resolve(intents);
                } catch (error) {
                    console.error('Failed to parse intents JSON:', error);
                    resolve([]);
                }
            });
  
            python.on('error', (error) => {
                console.error(error);
                reject(error);
            });
        } catch (error) {
            console.error(error);
            reject(error);
        }
    });
};
  
app.post('/api/classify', async (req, res) => {
    try {
        const { input, model } = req.body;
        const intents = await getIntents(input, model);
        const response = {
            intents: intents
        };
        res.json(response);
    } catch (error) {
        console.error(error);
        res.status(500).send('An error occurred');
    }
});
  
  
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});

