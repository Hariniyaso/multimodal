// Simplified responsive interaction logic

function toggleCamera(action) {
    fetch('/toggle_camera', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: action })
    })
    .then(response => response.json())
    .then(data => console.log('Camera state:', data.status))
    .catch(console.error);
}

function setMode(mode) {
    fetch('/set_mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: mode })
    })
    .then(response => response.json())
    .then(data => console.log('ML Module Output:', data.status))
    .catch(console.error);
}

function setLanguage(lang) {
    // Transmit requested localization to Backend server route
    fetch('/set_language', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ language: lang })
    })
    .then(response => response.json())
    .then(data => console.log('Language changed:', data.status))
    .catch(console.error);
}

function fetchOutput() {
    // 500ms polling allows rapid responsiveness for predictions
    fetch('/get_output')
        .then(res => res.json())
        .then(data => {
            if (data.output) {
                document.getElementById('ai-output').innerText = data.output;
            }
        })
        .catch(console.error);
}

// Start continuous looping
setInterval(fetchOutput, 500);
