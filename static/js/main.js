document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('drawing-canvas');
    const ctx = canvas.getContext('2d');
    const predictBtn = document.getElementById('predict-btn');
    const clearBtn = document.getElementById('clear-btn');
    const predictedDigit = document.getElementById('predicted-digit');

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    // Set initial canvas style
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    function draw(e) {
        if (!isDrawing) return;
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
        [lastX, lastY] = [e.offsetX, e.offsetY];
    }

    canvas.addEventListener('mousedown', (e) => {
        isDrawing = true;
        [lastX, lastY] = [e.offsetX, e.offsetY];
    });

    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', () => isDrawing = false);
    canvas.addEventListener('mouseout', () => isDrawing = false);

    clearBtn.addEventListener('click', () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        predictedDigit.textContent = '-';
    });

    predictBtn.addEventListener('click', async () => {
        const imageDataURL = canvas.toDataURL('image/png');
        
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageDataURL }),
        });

        const data = await response.json();
        predictedDigit.textContent = data.digit;
    });
});
