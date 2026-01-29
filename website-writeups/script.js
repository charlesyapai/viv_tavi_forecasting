document.addEventListener('DOMContentLoaded', () => {
    const params = new URLSearchParams(window.location.search);
    const configPath = params.get('config');

    if (!configPath) {
        document.getElementById('experiment-title').textContent = "No Configuration Selected";
        document.getElementById('experiment-description').innerHTML = "Please specify a config file in the URL, e.g., <br><code>viewer.html?config=configs/singapore.json</code>";
        return;
    }

    fetch(configPath)
        .then(response => {
            if (!response.ok) throw new Error(`Failed to load config: ${response.statusText}`);
            return response.json();
        })
        .then(config => {
            renderPage(config);
        })
        .catch(error => {
            console.error(error);
            document.getElementById('experiment-title').textContent = "Error Loading Config";
            document.getElementById('experiment-description').textContent = error.message;
        });
});

function renderPage(config) {
    document.title = config.title || "Experiment Results";
    document.getElementById('experiment-title').textContent = config.title || "Untitled Experiment";
    document.getElementById('experiment-description').textContent = config.description || "";

    const contentArea = document.getElementById('content-area');
    const basePath = config.basePath || "";

    config.sections.forEach(section => {
        const block = document.createElement('div');
        block.className = 'result-block';

        const headerDiv = document.createElement('div');
        headerDiv.className = 'block-header';
        const h2 = document.createElement('h2');
        h2.textContent = section.header;
        headerDiv.appendChild(h2);
        block.appendChild(headerDiv);

        const contentDiv = document.createElement('div');
        contentDiv.className = 'block-content';

        if (section.image) {
            const img = document.createElement('img');
            // Handle absolute vs relative paths logic if needed, but assuming relative to site root or absolute
            // If basePath is provided, prepend it
            img.src = basePath ? `${basePath}/${section.image}` : section.image;
            img.className = 'result-image';
            img.alt = section.header;
            contentDiv.appendChild(img);
        }

        if (section.caption) {
            const caption = document.createElement('div');
            caption.className = 'caption';
            caption.textContent = section.caption;
            contentDiv.appendChild(caption);
        }

        block.appendChild(contentDiv);
        contentArea.appendChild(block);
    });
}
