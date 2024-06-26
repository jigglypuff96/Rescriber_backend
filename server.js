const express = require("express");
const bodyParser = require("body-parser");
// Correctly import ollama
const { default: ollama } = require("ollama");
const ollamaModel = "llama3";
console.log(ollama);
const detectPrompt =
  'You are an expert in cybersecurity and data privacy. You are now tasked to detect PII from the given text, using the following taxonomy only:\n\nADDRESS\nIP_ADDRESS\nURL\nSSN\nPHONE_NUMBER\nEMAIL\nDRIVERS_LICENSE\nPASSPORT_NUMBER\nTAXPAYER IDENTIFICATION NUMBER\nID_NUMBER\nNAME\nUSERNAME\n\nGEOLOCATION: Places and locations, such as cities, provinces, countries, international regions, or named infrastructures (bus stops, bridges, etc.).\nAFFILIATION: Names of organizations, such as public and private companies, schools, universities, public institutions, prisons, healthcare institutions, non-governmental organizations, churches, etc.\nDEMOGRAPHIC_ATTRIBUTE: Demographic attributes of a person, such as native language, descent, heritage, ethnicity, nationality, religious or political group, birthmarks, ages, sexual orientation, gender and sex.\nTIME: Description of a specific date, time, or duration.\nHEALTH_INFORMATION: Details concerning an individual\'s health status, medical conditions, treatment records, and health insurance information.\nFINANCIAL_INFORMATION: Financial details such as bank account numbers, credit card numbers, investment records, salary information, and other financial statuses or activities.\nEDUCATIONAL_RECORD: Educational background details, including academic records, transcripts, degrees, and certification.\n\nFor the given message that a user sends to a chatbot, identify all the personally identifiable information using the above taxonomy only, and the entity_type should be selected from the all-caps categories.\nNote that the information should be related to a real person not in a public context, but okay if not uniquely identifiable.\nResult should be in its minimum possible unit.\nReturn me ONLY a json in the following format: {"results": [{"entity_type": YOU_DECIDE_THE_PII_TYPE, "text": PART_OF_MESSAGE_YOU_IDENTIFIED_AS_PII]}';

const app = express();
const port = 3000;

app.use(bodyParser.json());

app.post("/detect", async (req, res) => {
  const userMessage = req.body.message;
  console.log("userMessage is = ", userMessage);
  const systemPrompt = req.body.systemPrompt;

  try {
    // Check if the correct method is used, for example, it might be `ollama.chat` or something else
    const response = await ollama.chat({
      model: ollamaModel, // specify the model you want to use
      messages: [{ role: "user", content: userMessage }],
    });

    console.log("Ollama response:", response); // Log the full response for debugging
    res.send({ results: response.message.content });
  } catch (error) {
    console.error("Error running Ollama:", error);
    res
      .status(500)
      .send({ error: "Error running Ollama", details: error.message });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});

app.get("/", async (req, res) => {
  res.send("HI");
});

async function pullModel(modelName) {
  try {
    await ollama.pull({ model: modelName });
    console.log(`Model '${modelName}' has been pulled successfully.`);
  } catch (error) {
    console.error(`Failed to pull the model: ${error.message}`);
  }
}

async function chatWithOllama(systemPrompt, userMessage) {
  const modelName = "llama2";

  // Pull the model before using it
  console.log("pulling model");
  await pullModel(modelName);

  // Now proceed with the chat request
  const response = await ollama.chat({
    model: modelName,
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: userMessage },
    ],
    format: "json", // Set the expected format to JSON
    stream: false, // Set to false to get the full response at once
    // keep_alive and options are omitted since they are optional
  });
  console.log("response is", response);

  // Access and return the content of the response message
  return response.message.content;
}

// Example usage:

// pullModel(ollamaModel);
pullModel(ollamaModel).then(async () => {
  const modelfile = `
    FROM llama3
    SYSTEM "${detectPrompt}"
    `;
  const ollamaDetectModel = await ollama.create({
    model: "detectModel",
    modelfile: modelfile,
  });
  console.log("finish creating detectModel");
  console.log(ollamaDetectModel);
});
// chatWithOllama(systemPrompt, userMessage).then((response) => {
//   console.log(response);
// });
