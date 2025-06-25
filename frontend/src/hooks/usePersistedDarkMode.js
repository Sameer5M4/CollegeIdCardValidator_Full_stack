// src/hooks/usePersistedDarkMode.js

import { useState, useEffect } from 'react';

const usePersistedDarkMode = () => {
  const [isDarkMode, setIsDarkMode] = useState(() => {
    if (typeof window === 'undefined') return true;
    const storedPreference = window.localStorage.getItem('darkMode');
    if (storedPreference !== null) {
      return JSON.parse(storedPreference);
    }
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? true;
  });

  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDarkMode);
    try {
      window.localStorage.setItem('darkMode', JSON.stringify(isDarkMode));
    } catch (e) {
      console.error("Could not save dark mode preference to localStorage.", e);
    }
  }, [isDarkMode]);

  return [isDarkMode, setIsDarkMode];
};

export default usePersistedDarkMode;