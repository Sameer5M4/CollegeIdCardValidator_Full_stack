// src/lib/animations.js

export const statusBorderColors = {
    approved: "rgba(34, 197, 94, 0.5)",
    rejected: "rgba(239, 68, 68, 0.6)",
    manual_review: "rgba(245, 158, 11, 0.6)",
  };
  
  export const defaultBorderColor = "rgba(100, 116, 139, 0.2)";
  
  export const cardVariants = {
    hidden: { opacity: 0, y: 30, borderColor: defaultBorderColor },
    visible: (custom) => ({
      opacity: 1,
      y: 0,
      borderColor: custom ? statusBorderColors[custom.status] : defaultBorderColor,
      transition: {
        duration: 0.8,
        ease: 'easeOut',
      },
    }),
  };
  
  export const resultCardVariants = {
    hidden: { opacity: 0, scale: 0.95 },
    visible: {
      opacity: 1,
      scale: 1,
      transition: {
        duration: 0.4,
        ease: 'easeOut',
        staggerChildren: 0.1,
      },
    },
  };
  
  export const resultItemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeOut' } },
  };