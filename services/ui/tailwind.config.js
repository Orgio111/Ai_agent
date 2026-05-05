/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        jarvis: {
          primary: '#06b6d4',
          dark: '#030712',
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}
