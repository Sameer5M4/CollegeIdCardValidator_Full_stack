// src/components/DarkModeToggle.jsx

import { motion, AnimatePresence } from 'framer-motion';
import { Moon, Sun } from 'lucide-react';

const DarkModeToggle = ({ isDarkMode, setIsDarkMode }) => (
  <motion.button
    onClick={() => setIsDarkMode(!isDarkMode)}
    whileHover={{ scale: 1.1, rotate: 15 }}
    whileTap={{ scale: 0.9, rotate: -15 }}
    className="fixed z-50 top-5 left-5 p-3 rounded-full bg-slate-800/50 backdrop-blur-sm text-slate-300"
    aria-label="Toggle dark mode"
  >
    <AnimatePresence mode="wait" initial={false}>
      <motion.div
        key={isDarkMode ? 'moon' : 'sun'}
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        exit={{ y: 20, opacity: 0 }}
        transition={{ duration: 0.2 }}
      >
        {isDarkMode ? <Sun size={22} /> : <Moon size={22} />}
      </motion.div>
    </AnimatePresence>
  </motion.button>
);

export default DarkModeToggle;