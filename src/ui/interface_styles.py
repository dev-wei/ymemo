"""CSS styles and JavaScript for the UI interface."""

# Main CSS styles for the Voice Meeting App
APP_CSS = """
    .gradio-container {
        max-width: 1400px !important; 
        margin-left: auto !important;
        margin-right: auto !important;
        padding: 20px !important;
    }
    .header-text {
        text-align: center;
        margin-bottom: 20px;
    }
    
    
    /* Desktop Layout - Default */
    .meeting-row {
        margin-bottom: 20px !important;
    }
    .meeting-panel {
        height: 300px !important;
        overflow-y: auto !important;
        padding-right: 10px !important;
    }
    .main-content-row {
        gap: 1rem !important;
    }
    .dialog-panel {
        height: 1000px !important;
        overflow-y: auto !important;
        padding-left: 10px !important;
        padding-right: 10px !important;
    }
    .control-panel {
        height: 500px !important;
        overflow-y: auto !important;
        padding-left: 10px !important;
    }
    
    /* Mobile/Narrow Screen Layout */
    @media screen and (max-width: 768px) {
        .gradio-container {
            padding: 10px !important;
        }
        .meeting-panel {
            height: 250px !important;
        }
        .dialog-panel {
            height: 750px !important;
        }
        .control-panel {
            height: 400px !important;
        }
        
        
        
        /* Force vertical stacking with multiple selectors */
        .main-content-row,
        .main-content-row > .gradio-column,
        .main-content-row > div {
            display: flex !important;
            flex-direction: column !important;
            width: 100% !important;
        }
        
        /* Target Gradio's generated structure */
        .main-content-row > div:first-child,
        .main-content-row > div:last-child {
            width: 100% !important;
            max-width: 100% !important;
            flex: 1 1 100% !important;
            margin-bottom: 20px !important;
        }
        
        /* Override Gradio's default flex behavior */
        .gradio-row.main-content-row {
            flex-direction: column !important;
            align-items: stretch !important;
        }
        
        /* Force column layout */
        .gradio-column {
            width: 100% !important;
            max-width: 100% !important;
            flex-basis: 100% !important;
        }
    }
    
    /* Very narrow screens (mobile portrait) */
    @media screen and (max-width: 480px) {
        .gradio-container {
            padding: 5px !important;
        }
        .meeting-panel {
            height: 200px !important;
        }
        .dialog-panel {
            height: 350px !important;
        }
        .control-panel {
            height: 350px !important;
        }
        
        /* Ensure vertical stacking on very small screens */
        .main-content-row,
        .gradio-row.main-content-row {
            flex-direction: column !important;
            align-items: stretch !important;
        }
        
        .main-content-row > div,
        .main-content-row > .gradio-column {
            width: 100% !important;
            max-width: 100% !important;
            flex: 1 1 100% !important;
            margin-bottom: 15px !important;
        }
    }
    
    /* Prevent horizontal scrolling issues */
    .gradio-row {
        gap: 1rem !important;
    }
    
    /* Ensure proper box sizing */
    * {
        box-sizing: border-box !important;
    }
    
    /* Additional mobile layout fixes */
    @media screen and (max-width: 768px) {
        /* Force all direct children of main-content-row to be full width */
        .main-content-row > * {
            width: 100% !important;
            max-width: 100% !important;
        }
        
        /* Override any inline styles that might prevent stacking */
        .main-content-row [style*="width"] {
            width: 100% !important;
        }
        
        /* Ensure Gradio's column system respects mobile layout */
        .gradio-row.main-content-row > .gradio-column {
            flex: 1 1 100% !important;
            width: 100% !important;
            max-width: 100% !important;
        }
    }
    
    /* Responsive improvements */
    @media screen and (max-width: 768px) {
        h1, h2, h3 {
            font-size: 1.2em !important;
        }
        button {
            padding: 8px 16px !important;
            font-size: 0.9em !important;
        }
    }
"""

# JavaScript for theme switching
APP_JS = """
    function refresh() {
        const url = new URL(window.location);

        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
"""