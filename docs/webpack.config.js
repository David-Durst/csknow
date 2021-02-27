// Import path for resolving file paths
var path = require("path");
module.exports = {
    // Specify the entry point for our app.
    entry: [path.join(__dirname, "src", "browser.ts")],
    devtool: 'inline-source-map',
    // Specify the output file containing our bundled code.
    output: {
        path: path.resolve(__dirname, 'dist'),
        filename: 'bundle.js',
        library: "exposed"
    },
    module: {
        rules: [
            {
                test: /\.tsx?$/,
                use: 'ts-loader',
                exclude: /node_modules/,
            },
        ],
    },
    resolve:{
        extensions: [ '.tsx', '.ts', '.js' ],
    }
    /**
     * In Webpack version v2.0.0 and earlier, you must tell
     * webpack how to use "json-loader" to load 'json' files.
     * To do this Enter 'npm --save-dev install json-loader' at the
     * command line to install the "json-loader' package, and include the
     * following entry in your webpack.config.js.
     module: {
    rules: [{test: /\.json$/, use: use: "json-loader"}]
  }
     **/
};