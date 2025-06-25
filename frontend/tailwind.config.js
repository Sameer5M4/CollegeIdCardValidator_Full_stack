// tailwind.config.js

/** @type {import('tailwindcss').Config} */
export default {
    content: [
      "./index.html",
      "./src/**/*.{js,ts,jsx,tsx}",
    ],
    // THIS IS THE LINE YOU MUST ADD OR CHANGE:
    darkMode: 'class',
    //
    theme: {
      extend: {
          // Your other theme extensions might be here
      },
    },
    plugins: [],
  }