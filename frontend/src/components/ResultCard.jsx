// src/components/ResultCard.jsx

import { motion } from 'framer-motion';
import { ShieldCheck, ShieldAlert, ShieldQuestion } from 'lucide-react';
import { resultCardVariants, resultItemVariants } from '../lib/animations';

const ResultCard = ({ result }) => {
  const statusConfig = {
    approved: { icon: <ShieldCheck className="w-20 h-20 text-green-400" />, title: 'Approved', textColor: 'text-green-400', borderColor: 'border-green-500/30', bgColor: 'bg-green-500/10' },
    rejected: { icon: <ShieldAlert className="w-20 h-20 text-red-400" />, title: 'Rejected', textColor: 'text-red-400', borderColor: 'border-red-500/30', bgColor: 'bg-red-500/10' },
    manual_review: { icon: <ShieldQuestion className="w-20 h-20 text-amber-400" />, title: 'Manual Review', textColor: 'text-amber-400', borderColor: 'border-amber-500/30', bgColor: 'bg-amber-500/10' },
  };
  const config = statusConfig[result.status] || statusConfig.manual_review;
  const scorePercentage = result.validation_score * 100;
  const circumference = 2 * Math.PI * 52;

  return (
    <motion.div variants={resultCardVariants} className={`w-full max-w-md p-8 rounded-2xl ${config.bgColor} border ${config.borderColor} shadow-lg backdrop-blur-sm`}>
      <motion.div variants={resultItemVariants} className="flex flex-col items-center text-center">
        {config.icon}
        <h2 className={`text-4xl font-bold mt-4 ${config.textColor}`}>{config.title}</h2>
        <p className="text-slate-400 mt-1">ID Validation Complete</p>
      </motion.div>
      <motion.div variants={resultItemVariants} className="relative flex justify-center items-center my-10">
        <svg className="w-32 h-32 transform -rotate-90" viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="52" stroke="currentColor" strokeWidth="12" className="text-slate-700/50" fill="transparent" />
          <motion.circle cx="60" cy="60" r="52" stroke="currentColor" strokeWidth="12" className={config.textColor} fill="transparent" strokeLinecap="round" strokeDasharray={circumference} initial={{ strokeDashoffset: circumference }} animate={{ strokeDashoffset: circumference - (scorePercentage / 100) * circumference }} transition={{ duration: 1.5, ease: 'circOut', delay: 0.5 }} />
        </svg>
        <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.7 }} className="absolute text-3xl font-bold text-slate-100">{`${Math.round(scorePercentage)}%`}</motion.span>
      </motion.div>
      <motion.div variants={resultItemVariants} className="space-y-4 text-base">
        <div className="flex justify-between items-center"><span className="text-slate-400">Label:</span><span className="font-semibold text-slate-100 capitalize px-3 py-1 bg-slate-700/50 rounded-full">{result.label}</span></div>
        <hr className="border-slate-700/50" />
        <div className="flex justify-between items-start"><span className="text-slate-400">Reason:</span><span className="font-semibold text-slate-100 text-right max-w-[70%]">{result.reason}</span></div>
        <hr className="border-slate-700/50" />
        <div className="flex justify-between items-center"><span className="text-slate-400">User ID:</span><span className="font-mono text-slate-100">{result.user_id}</span></div>
      </motion.div>
    </motion.div>
  );
};

export default ResultCard;