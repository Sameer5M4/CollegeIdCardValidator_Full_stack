// src/components/ThematicLoadingIndicator.jsx

import { motion } from 'framer-motion';

const ThematicLoadingIndicator = () => (
    <motion.div
        key="loader"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="flex flex-col items-center justify-center w-full"
    >
        <div className="w-80 h-48 bg-slate-800/50 rounded-xl flex items-center p-4 gap-4 relative overflow-hidden border border-slate-700">
            <div className="w-20 h-28 bg-slate-700/60 rounded-md flex-shrink-0"></div>
            <div className="flex-1 space-y-3">
                <div className="w-full h-4 bg-slate-700/60 rounded-full"></div>
                <div className="w-5/6 h-4 bg-slate-700/60 rounded-full"></div>
                <div className="w-full h-4 bg-slate-700/60 rounded-full"></div>
                <div className="w-3/4 h-4 bg-slate-700/60 rounded-full"></div>
            </div>
            <motion.div
                className="absolute top-0 left-0 right-0 h-1 bg-cyan-400"
                style={{ boxShadow: '0 0 10px 2px #22d3ee' }}
                animate={{ y: [0, 192] }}
                transition={{ duration: 1.5, repeat: Infinity, repeatType: 'reverse', ease: 'easeInOut' }}
            />
        </div>
        <p className="text-slate-400 mt-6 text-lg tracking-wider">
            Analyzing Authenticity...
        </p>
    </motion.div>
);

export default ThematicLoadingIndicator;