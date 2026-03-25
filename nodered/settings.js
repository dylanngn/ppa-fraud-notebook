/**
 * Node-RED settings for the PPA Fraud Detection demo.
 *
 * Dashboard:   http://localhost:1880/dashboard
 * Flow editor: http://localhost:1880/
 */
module.exports = {
    uiPort: process.env.PORT || 1880,

    // Store flows in this directory (next to this file)
    userDir: __dirname,

    // Flow file
    flowFile: 'flows.json',
    flowFilePretty: true,

    // Dashboard path
    ui: { path: 'dashboard' },

    // Logging
    logging: {
        console: {
            level: 'info',
            metrics: false,
            audit: false,
        },
    },

    // Editor theme
    editorTheme: {
        page: {
            title: 'PPA Fraud Detection Pipeline',
        },
        header: {
            title: 'Fraud Detection — Node-RED',
        },
    },

    // Allow function nodes to access external modules
    functionExternalModules: true,

    // Context storage (in-memory only — sufficient for demo)
    contextStorage: {
        default: { module: 'memory' },
    },

    // Fixed credential secret suppresses the "system-generated key" warning.
    // No actual credentials are stored in this demo flow.
    credentialSecret: 'ppa-fraud-demo-secret',
}
