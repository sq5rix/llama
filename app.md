Okay, let's create that simple webpage structure.
This example uses basic HTML, CSS, and JavaScript. It simulates the process. In a real application, clicking the "execute" arrow would trigger a JavaScript function to make an API call to the Gemini API (likely via a backend server you'd also need to create to securely handle your API key).
HTML (index.html):
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gemini Prompt Interface</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>

    <h1>Gemini Prompt Examples</h1>

    <div class="examples-container">
        <div class="example" id="example1">
            <h2>Example 1: Creative Writing</h2>
            <p>Write a short story about a robot who discovers music for the first time.</p>
        </div>
        <div class="example" id="example2">
            <h2>Example 2: Code Generation</h2>
            <p>Generate a Python function that takes a list of numbers and returns the sum of all even numbers in the list.</p>
        </div>
    </div>

    <h2>Your Prompt</h2>
    <div class="input-container">
        <textarea id="prompt-input" placeholder="Type your prompt here, or click an example above..."></textarea>
        <button id="execute-button" title="Execute Prompt">&#10148;</button> </div>

    <div id="output-area" style="margin-top: 20px; border: 1px dashed #ccc; padding: 10px; min-height: 50px; background-color: #f9f9f9;">
        (Simulated API response will appear here)
    </div>

    <script src="script.js"></script>
</body>
</html>

CSS (style.css):
body {
    font-family: sans-serif;
    line-height: 1.6;
    margin: 20px;
    background-color: #f4f4f4;
    color: #333;
}

h1, h2 {
    color: #444;
}

.examples-container {
    display: flex;
    gap: 20px; /* Space between example boxes */
    margin-bottom: 30px;
}

.example {
    flex: 1; /* Each example takes equal width */
    border: 1px solid #ccc;
    padding: 15px;
    border-radius: 5px;
    background-color: #fff;
    cursor: pointer;
    transition: box-shadow 0.2s ease-in-out;
}

.example:hover {
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.example h2 {
    margin-top: 0;
    font-size: 1.1em;
    color: #0056b3; /* Example title color */
}

.input-container {
    display: flex;
    align-items: stretch; /* Makes button same height as textarea */
    margin-top: 10px;
}

#prompt-input {
    flex-grow: 1; /* Textarea takes up available space */
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 5px 0 0 5px; /* Rounded corners only on the left */
    font-size: 1em;
    min-height: 80px;
    resize: vertical; /* Allow vertical resizing */
    margin-right: -1px; /* Overlap border slightly */
}

#execute-button {
    padding: 10px 20px;
    font-size: 1.5em; /* Make arrow bigger */
    cursor: pointer;
    border: 1px solid #0056b3;
    background-color: #007bff;
    color: white;
    border-radius: 0 5px 5px 0; /* Rounded corners only on the right */
    transition: background-color 0.2s ease;
}

#execute-button:hover {
    background-color: #0056b3;
}

#output-area {
    margin-top: 20px;
    border: 1px dashed #ccc;
    padding: 15px;
    min-height: 50px;
    background-color: #e9ecef; /* Light background for output */
    white-space: pre-wrap; /* Preserve whitespace/newlines in simulated output */
    font-family: monospace; /* Good for code/formatted text */
}

JavaScript (script.js):
document.addEventListener('DOMContentLoaded', () => {
    // --- Configuration ---
    // PASTE YOUR SYSTEM PROMPT FROM GOOGLE AI STUDIO HERE:
    const systemPrompt = `
You are a helpful AI assistant. Your goal is to provide accurate and concise information.
Format your answers clearly.
    `.trim(); // .trim() removes leading/trailing whitespace

    // --- Get DOM Elements ---
    const example1 = document.getElementById('example1');
    const example2 = document.getElementById('example2');
    const promptInput = document.getElementById('prompt-input');
    const executeButton = document.getElementById('execute-button');
    const outputArea = document.getElementById('output-area');

    // --- Event Listeners ---

    // Add click listeners to example prompts
    if (example1) {
        example1.addEventListener('click', () => {
            // Find the <p> tag within the clicked example and get its text
            const promptText = example1.querySelector('p')?.textContent;
            if (promptText) {
                promptInput.value = promptText.trim();
                 outputArea.textContent = '(Prompt loaded from Example 1)';
            }
        });
    }

    if (example2) {
         example2.addEventListener('click', () => {
            const promptText = example2.querySelector('p')?.textContent;
            if (promptText) {
                promptInput.value = promptText.trim();
                outputArea.textContent = '(Prompt loaded from Example 2)';
            }
        });
    }

    // Add click listener to the execute button
    if (executeButton) {
        executeButton.addEventListener('click', () => {
            const userPrompt = promptInput.value.trim();

            if (!userPrompt) {
                outputArea.textContent = 'Please enter a prompt or select an example.';
                return; // Stop if the prompt is empty
            }

            // ** SIMULATION AREA **
            // In a real application, you would send the 'systemPrompt' and 'userPrompt'
            // to your backend server, which would then call the Gemini API.
            // For now, we just display what *would* be sent.

            console.log("--- Sending to API (Simulated) ---");
            console.log("System Prompt:", systemPrompt);
            console.log("User Prompt:", userPrompt);
            console.log("------------------------------------");

            // Simulate receiving a response
            outputArea.textContent = `Simulating API Call...
System Prompt was: "${systemPrompt}"
User Prompt was: "${userPrompt}"

(Replace this with actual API call logic and response handling)`;

            // You would add your fetch() call here to your backend endpoint
            // fetch('/api/gemini', {
            //     method: 'POST',
            //     headers: { 'Content-Type': 'application/json' },
            //     body: JSON.stringify({ systemPrompt: systemPrompt, userPrompt: userPrompt })
            // })
            // .then(response => response.json())
            // .then(data => {
            //     outputArea.textContent = data.responseText; // Or however your API returns it
            // })
            // .catch(error => {
            //     console.error('Error calling API:', error);
            //     outputArea.textContent = 'Error communicating with the API.';
            // });
        });
    }
});

How to Use:
 * Save the HTML code as index.html.
 * Save the CSS code as style.css in the same folder.
 * Save the JavaScript code as script.js in the same folder.
 * Crucially: Open script.js and replace the placeholder text inside the systemPrompt variable with the actual system prompt you copied from your Gemini setup in Google AI Studio.
 * Open the index.html file in your web browser.
You will see the two example prompts side-by-side. Clicking one will copy its text into the text area below. Typing in the text area and clicking the arrow button will simulate the process by logging the combined prompts to the browser's developer console and showing a message in the output area on the page.
