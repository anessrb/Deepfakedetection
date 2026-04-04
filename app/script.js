document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const mediaPreviewBox = document.getElementById('media-preview-box');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resetBtn = document.getElementById('reset-btn');
    const resultsSection = document.getElementById('results-section');
    const loader = document.getElementById('loader');

    const resultBadge = document.getElementById('result-badge');
    const scoreFill = document.getElementById('score-fill');
    const scoreText = document.getElementById('score-text');

    const visualTitle = document.getElementById('visual-title');
    const visualInfo = document.getElementById('visual-info');
    const gradcamImage = document.getElementById('gradcam-image');
    const videoResults = document.getElementById('video-results');
    const temporalCanvas = document.getElementById('temporal-chart');

    let selectedFile = null;
    let chart = null;

    const API_BASE = 'http://localhost:8000';

    // Drag and drop handlers
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        const isImage = file.type.startsWith('image/');
        const isVideo = file.type.startsWith('video/');

        if (!isImage && !isVideo) {
            alert('Please upload an image or video file.');
            return;
        }

        selectedFile = file;
        const reader = new FileReader();

        reader.onload = (e) => {
            mediaPreviewBox.innerHTML = '';
            if (isImage) {
                const img = document.createElement('img');
                img.src = e.target.result;
                img.id = 'image-preview';
                mediaPreviewBox.appendChild(img);
            } else {
                const video = document.createElement('video');
                video.src = e.target.result;
                video.controls = true;
                video.id = 'video-preview';
                mediaPreviewBox.appendChild(video);
            }

            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            resultsSection.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    resetBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        dropZone.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        resultsSection.classList.add('hidden');
        if (chart) {
            chart.destroy();
            chart = null;
        }
    });

    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        loader.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch(`${API_BASE}/predict`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Analysis failed');
            }

            const data = await response.json();
            displayResults(data);
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during analysis. Please make sure the backend is running.');
        } finally {
            loader.classList.add('hidden');
        }
    });

    function displayResults(data) {
        resultsSection.classList.remove('hidden');

        // Update badge
        resultBadge.textContent = data.decision;
        resultBadge.className = 'badge ' + data.decision.toLowerCase();

        // Update score
        const probPercent = (data.probability * 100).toFixed(2);
        scoreText.textContent = probPercent + '%';
        scoreFill.style.width = probPercent + '%';
        scoreFill.style.backgroundColor = data.decision === 'FAKE' ? 'var(--danger)' : 'var(--success)';

        if (data.type === 'image') {
            // Image specific results
            visualTitle.textContent = 'AI Attention Map (Grad-CAM)';
            visualInfo.textContent = 'Heatmap shows regions where the AI detected manipulation traces.';
            videoResults.classList.add('hidden');

            if (data.gradcam_url) {
                gradcamImage.src = API_BASE + data.gradcam_url;
                gradcamImage.classList.remove('hidden');
            } else {
                gradcamImage.classList.add('hidden');
            }
        } else {
            // Video specific results
            visualTitle.textContent = 'Temporal Analysis';
            visualInfo.textContent = 'Aggregated decision based on multiple frames across the video.';
            gradcamImage.classList.add('hidden');
            videoResults.classList.remove('hidden');

            renderChart(data.frame_probs);
        }

        // Smooth scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function renderChart(probs) {
        if (chart) {
            chart.destroy();
        }

        const ctx = temporalCanvas.getContext('2d');
        const labels = probs.map((_, i) => `Frame ${i + 1}`);

        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Fake Probability',
                    data: probs,
                    borderColor: '#58A6FF',
                    backgroundColor: 'rgba(88, 166, 255, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointBackgroundColor: '#58A6FF'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#8B949E'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#8B949E',
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(22, 27, 34, 0.9)',
                        titleColor: '#fff',
                        bodyColor: '#8B949E',
                        borderColor: '#30363D',
                        borderWidth: 1
                    }
                }
            }
        });
    }
});
