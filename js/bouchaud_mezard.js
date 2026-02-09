async function runSimulation() {
    const statusText = document.getElementById('status');
    const imgElement = document.getElementById('simulation-plot');
    const button = document.getElementById('run-btn');

    // Find parameter values (you can add inputs in the HTML)
    const params = {
        N: parseInt(document.getElementById('param-N').value) || 500,
        J: parseFloat(document.getElementById('param-J').value) || 0.5,
        sigma: parseFloat(document.getElementById('param-sigma').value) || 0.2,
        graph_type: document.getElementById('param-graph').value || "random_regular",
        k: parseInt(document.getElementById('param-k').value) || 5,
        steps: parseInt(document.getElementById('param-steps').value) || 1000
    };

    statusText.innerText = "Simulation in progress... please wait.";
    button.disabled = true;

    try {
        const response = await fetch('https://mattiatara-bouchaud-mezard.hf.space/simulate/plot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });

        if (!response.ok) throw new Error("API Error: " + response.statusText);

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        imgElement.src = url;
        imgElement.style.display = "block";
        statusText.innerText = "Simulation completed!";
    } catch (error) {
        statusText.innerText = "Error: " + error.message;
    } finally {
        button.disabled = false;
    }
}