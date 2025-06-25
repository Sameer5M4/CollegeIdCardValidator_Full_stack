// import { useState, useCallback } from 'react';
// import { motion, AnimatePresence } from 'framer-motion';
// import { UploadCloud, AlertTriangle } from 'lucide-react';

// // Local Imports
// import './App.css'
// import { cardVariants } from './lib/animations';
// import usePersistedDarkMode from './hooks/usePersistedDarkMode';
// import DarkModeToggle from './components/DarkModeToggle';
// import ImagePreview from './components/ImagePreview';
// import ResultCard from './components/ResultCard';
// import ThematicLoadingIndicator from './components/ThematicLoadingIndicator';
// import InitialPlaceholder from './components/InitialPlaceholder';

// // --- Configuration ---
// // const API_URL = 'http://localhost:8000/validate-id';
// // const API_URL = 'http://backend:5000/validate-id';
// // --- Configuration ---
// const API_URL = 'http://localhost:5000/validate-id'; ✅



// function App() {
//   const [userId, setUserId] = useState('stu_2290');
//   const [imageBase64, setImageBase64] = useState(null);
//   const [imagePreview, setImagePreview] = useState(null);
//   const [result, setResult] = useState(null);
//   const [isLoading, setIsLoading] = useState(false);
//   const [error, setError] = useState('');

//   const [isDarkMode, setIsDarkMode] = usePersistedDarkMode();

//   const handleImageChange = (e) => {
//     const file = e.target.files[0];
//     if (file && (file.type === "image/jpeg" || file.type === "image/png")) {
//       const reader = new FileReader();
//       reader.onloadend = () => {
//         setImageBase64(reader.result);
//         setImagePreview(URL.createObjectURL(file));
//         setError('');
//         setResult(null);
//       };
//       reader.readAsDataURL(file);
//     } else {
//       setError('Please upload a valid image file (JPEG or PNG).');
//       setImageBase64(null);
//       setImagePreview(null);
//     }
//   };

//   const handleReset = useCallback(() => {
//     setUserId('stu_2290');
//     setImageBase64(null);
//     if (imagePreview) URL.revokeObjectURL(imagePreview);
//     setImagePreview(null);
//     setResult(null);
//     setError('');
//     setIsLoading(false);
//     const fileInput = document.getElementById('file-upload');
//     if (fileInput) fileInput.value = '';
//   }, [imagePreview]);

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     if (!imageBase64 || !userId) {
//       setError('User ID and an image are required.');
//       return;
//     }
//     setIsLoading(true);
//     setError('');
//     setResult(null);

//     try {
//       await new Promise(resolve => setTimeout(resolve, 2500)); // Simulate network delay
//       const response = await fetch(API_URL, {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ user_id: userId, image_base64: imageBase64 }),
//       });
//       if (!response.ok) {
//         const errorData = await response.json().catch(() => ({}));
//         throw new Error(errorData.detail || `Server error: ${response.status}`);
//       }
//       const data = await response.json();
//       setResult(data);
//     } catch (err) {
//       setError(err.message || 'An unexpected error occurred. Please try again.');
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   return (
//     <div className="min-h-screen w-full font-sans bg-gray-100 dark:bg-gray-900 dark:bg-radial-gradient from-gray-900 to-black transition-colors duration-300">
//       <DarkModeToggle isDarkMode={isDarkMode} setIsDarkMode={setIsDarkMode} />
//       <main className="min-h-screen flex items-center justify-center p-4 md:p-8">
//         <motion.div
//           variants={cardVariants}
//           initial="hidden"
//           animate="visible"
//           custom={result}
//           className="w-full max-w-6xl mx-auto bg-white/5 dark:bg-gray-800/20 rounded-3xl shadow-2xl backdrop-blur-2xl overflow-hidden md:grid md:grid-cols-5 border-2"
//         >
//           {/* Left Column: Form */}
//           <div className="p-8 md:p-12 md:col-span-2 space-y-8">
//             <header>
//               <h1 className="text-4xl lg:text-5xl font-bold bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">AI Validator</h1>
//               <p className="text-slate-400 mt-2">Verify ID authenticity with the power of AI.</p>
//             </header>
//             <form onSubmit={handleSubmit} className="space-y-6">
//               <div>
//                 <label htmlFor="user-id" className="block text-sm font-medium text-slate-300 mb-2">User ID</label>
//                 <input type="text" id="user-id" value={userId} onChange={(e) => setUserId(e.target.value)} className="w-full px-4 py-3 bg-gray-900/50 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all" placeholder="e.g., stu_2290" required />
//               </div>
//               <AnimatePresence mode="wait">
//                 {imagePreview ? (<ImagePreview image={imagePreview} onReset={handleReset} />) : (
//                   <motion.div key="upload" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
//                     <label htmlFor="file-upload" className="relative flex flex-col items-center justify-center w-full h-48 border-2 border-slate-700 border-dashed rounded-lg cursor-pointer bg-gray-900/50 hover:border-blue-500 hover:bg-gray-900 transition-all">
//                       <div className="text-center"><UploadCloud className="mx-auto h-12 w-12 text-slate-500" /><p className="mt-2 text-sm text-slate-300"><span className="font-semibold">Click to upload</span></p><p className="text-xs text-slate-500">PNG or JPG</p></div>
//                       <input id="file-upload" type="file" className="sr-only" onChange={handleImageChange} accept="image/png, image/jpeg" />
//                     </label>
//                   </motion.div>
//                 )}
//               </AnimatePresence>
//               {error && (<motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="flex items-center gap-2 text-red-400 text-sm p-3 bg-red-900/30 rounded-lg"><AlertTriangle size={16} /><span>{error}</span></motion.div>)}
//               <div className="flex flex-col sm:flex-row gap-4 pt-4">
//                 <motion.button type="submit" disabled={!imageBase64 || isLoading} whileHover={{ scale: 1.03, filter: 'brightness(1.1)', boxShadow: '0 10px 20px -5px rgba(59, 130, 246, 0.3)' }} whileTap={{ scale: 0.98, filter: 'brightness(0.9)' }} transition={{ duration: 0.2, ease: "circOut" }} className="w-full flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-lg text-white bg-blue-600 hover:bg-blue-700 disabled:bg-slate-500 disabled:shadow-none disabled:cursor-not-allowed transition-all">
//                   {isLoading ? 'Validating...' : 'Validate ID'}
//                 </motion.button>
//                 <motion.button type="button" onClick={handleReset} whileHover={{ scale: 1.05, filter: 'brightness(1.2)' }} whileTap={{ scale: 0.95, filter: 'brightness(0.9)' }} transition={{ duration: 0.2, ease: "circOut" }} className="w-full sm:w-auto px-6 py-3 border border-slate-600 text-base font-medium rounded-lg text-slate-300 hover:bg-slate-700 transition-colors">
//                   Reset
//                 </motion.button>
//               </div>
//             </form>
//           </div>
//           {/* Right Column: Results */}
//           <div className="bg-black/20 p-8 md:p-12 md:col-span-3 flex items-center justify-center min-h-[400px] md:min-h-full">
//             <AnimatePresence mode="wait">
//               {isLoading && <ThematicLoadingIndicator />}
//               {!isLoading && result && <ResultCard result={result} key="result" />}
//               {!isLoading && !result && <InitialPlaceholder key="placeholder" />}
//             </AnimatePresence>
//           </div>
//         </motion.div>
//       </main>
//     </div>
//   );
// }

// export default App;
import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, AlertTriangle } from 'lucide-react';

// Local Imports
import './App.css'
import { cardVariants } from './lib/animations';
// ✅ FIX: ADD THIS IMPORT LINE. THIS IS THE CAUSE OF THE ERROR.
import usePersistedDarkMode from './hooks/usePersistedDarkMode'; 
import DarkModeToggle from './components/DarkModeToggle';
import ImagePreview from './components/ImagePreview';
import ResultCard from './components/ResultCard';
import ThematicLoadingIndicator from './components/ThematicLoadingIndicator';
import InitialPlaceholder from './components/InitialPlaceholder';

// --- Configuration ---
const API_URL = 'http://localhost:5000/validate-id';

function App() {
  const [userId, setUserId] = useState('stu_2290');
  const [imageBase64, setImageBase64] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // This line was causing the error because the import was missing
  const [isDarkMode, setIsDarkMode] = usePersistedDarkMode();

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file && (file.type === "image/jpeg" || file.type === "image/png")) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImageBase64(reader.result);
        setImagePreview(URL.createObjectURL(file));
        setError('');
        setResult(null);
      };
      reader.readAsDataURL(file);
    } else {
      setError('Please upload a valid image file (JPEG or PNG).');
      setImageBase64(null);
      setImagePreview(null);
    }
  };

  const handleReset = useCallback(() => {
    setUserId('stu_2290');
    setImageBase64(null);
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    setImagePreview(null);
    setResult(null);
    setError('');
    setIsLoading(false);
    const fileInput = document.getElementById('file-upload');
    if (fileInput) fileInput.value = '';
  }, [imagePreview]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!imageBase64 || !userId) {
      setError('User ID and an image are required.');
      return;
    }
    setIsLoading(true);
    setError('');
    setResult(null);

    try {
      // await new Promise(resolve => setTimeout(resolve, 2500)); // Simulate network delay
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, image_base64: imageBase64 }),
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }
      const data = await response.json();
      setResult(data);
    } catch (err) {
      let errorMessage = err.message || 'An unexpected error occurred.';
      if (errorMessage.includes('Failed to fetch')) {
          errorMessage = 'Could not connect to the backend server. Is it running?';
      }
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  // The rest of your JSX is correct
  return (
    <div className="min-h-screen w-full font-sans bg-gray-100 dark:bg-gray-900 dark:bg-radial-gradient from-gray-900 to-black transition-colors duration-300">
      <DarkModeToggle isDarkMode={isDarkMode} setIsDarkMode={setIsDarkMode} />
      <main className="min-h-screen flex items-center justify-center p-4 md:p-8">
        <motion.div
          variants={cardVariants}
          initial="hidden"
          animate="visible"
          custom={result}
          className="w-full max-w-6xl mx-auto bg-white/5 dark:bg-gray-800/20 rounded-3xl shadow-2xl backdrop-blur-2xl overflow-hidden md:grid md:grid-cols-5 border-2"
        >
          {/* Left Column: Form */}
          <div className="p-8 md:p-12 md:col-span-2 space-y-8">
            <header>
              <h1 className="text-4xl lg:text-5xl font-bold bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">AI Validator</h1>
              <p className="text-slate-400 mt-2">Verify ID authenticity with the power of AI.</p>
            </header>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label htmlFor="user-id" className="block text-sm font-medium text-slate-300 mb-2">User ID</label>
                <input type="text" id="user-id" value={userId} onChange={(e) => setUserId(e.target.value)} className="w-full px-4 py-3 bg-gray-900/50 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all" placeholder="e.g., stu_2290" required />
              </div>
              <AnimatePresence mode="wait">
                {imagePreview ? (<ImagePreview image={imagePreview} onReset={handleReset} />) : (
                  <motion.div key="upload" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    <label htmlFor="file-upload" className="relative flex flex-col items-center justify-center w-full h-48 border-2 border-slate-700 border-dashed rounded-lg cursor-pointer bg-gray-900/50 hover:border-blue-500 hover:bg-gray-900 transition-all">
                      <div className="text-center"><UploadCloud className="mx-auto h-12 w-12 text-slate-500" /><p className="mt-2 text-sm text-slate-300"><span className="font-semibold">Click to upload</span></p><p className="text-xs text-slate-500">PNG or JPG</p></div>
                      <input id="file-upload" type="file" className="sr-only" onChange={handleImageChange} accept="image/png, image/jpeg" />
                    </label>
                  </motion.div>
                )}
              </AnimatePresence>
              {error && (<motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="flex items-center gap-2 text-red-400 text-sm p-3 bg-red-900/30 rounded-lg"><AlertTriangle size={16} /><span>{error}</span></motion.div>)}
              <div className="flex flex-col sm:flex-row gap-4 pt-4">
                <motion.button type="submit" disabled={!imageBase64 || isLoading} whileHover={{ scale: 1.03, filter: 'brightness(1.1)', boxShadow: '0 10px 20px -5px rgba(59, 130, 246, 0.3)' }} whileTap={{ scale: 0.98, filter: 'brightness(0.9)' }} transition={{ duration: 0.2, ease: "circOut" }} className="w-full flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-lg text-white bg-blue-600 hover:bg-blue-700 disabled:bg-slate-500 disabled:shadow-none disabled:cursor-not-allowed transition-all">
                  {isLoading ? 'Validating...' : 'Validate ID'}
                </motion.button>
                <motion.button type="button" onClick={handleReset} whileHover={{ scale: 1.05, filter: 'brightness(1.2)' }} whileTap={{ scale: 0.95, filter: 'brightness(0.9)' }} transition={{ duration: 0.2, ease: "circOut" }} className="w-full sm:w-auto px-6 py-3 border border-slate-600 text-base font-medium rounded-lg text-slate-300 hover:bg-slate-700 transition-colors">
                  Reset
                </motion.button>
              </div>
            </form>
          </div>
          {/* Right Column: Results */}
          <div className="bg-black/20 p-8 md:p-12 md:col-span-3 flex items-center justify-center min-h-[400px] md:min-h-full">
            <AnimatePresence mode="wait">
              {isLoading && <ThematicLoadingIndicator />}
              {!isLoading && result && <ResultCard result={result} key="result" />}
              {!isLoading && !result && <InitialPlaceholder key="placeholder" />}
            </AnimatePresence>
          </div>
        </motion.div>
      </main>
    </div>
  );
}

export default App;