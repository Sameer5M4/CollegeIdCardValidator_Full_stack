// src/components/InitialPlaceholder.jsx

import { motion } from 'framer-motion';
import { FileImage } from 'lucide-react';
import { resultItemVariants } from '../lib/animations';

const InitialPlaceholder = () => (
    <motion.div variants={resultItemVariants} className="flex flex-col items-center justify-center h-full text-center p-8 border-2 border-dashed border-slate-700/50 rounded-2xl">
        <FileImage className="w-20 h-20 text-slate-600" />
        <h3 className="mt-6 text-2xl font-semibold text-slate-200">Results will appear here</h3>
        <p className="mt-2 text-slate-400 max-w-xs">Upload an ID card image to begin the AI-powered validation process.</p>
    </motion.div>
);

export default InitialPlaceholder;