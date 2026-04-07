import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, Loader2, RefreshCw, Scan, Fingerprint } from 'lucide-react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

export default function Dashboard({ token }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [scanType, setScanType] = useState('xray');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResults(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('image', file);
    formData.append('scanType', scanType);

    try {
      const response = await axios.post(`${API_URL}/api/analyze`, formData, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setResults(response.data);
    } catch (error) {
      if(error.response?.status === 401) alert("Session expired. Please log in again.");
      else alert("Analysis failed. Check backend console.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-8 pt-28 pb-20">
      <motion.div 
        initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
        className="glass-panel rounded-3xl p-8 flex flex-col md:flex-row gap-8 items-stretch"
      >
        <div className="w-full md:w-1/3 flex flex-col justify-between space-y-6">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-400 text-xs font-semibold uppercase tracking-wider border border-cyan-500/20 mb-4">
              <Scan className="w-4 h-4" /> Inference Engine
            </div>
            <h2 className="text-3xl font-bold text-white mb-2">Upload Scan</h2>
            <p className="text-slate-400 text-sm leading-relaxed">Securely run patient radiographs through the triple-architecture XAI ensemble.</p>
          </div>
          
          <div className="space-y-4 mt-auto">
            <select 
              value={scanType} onChange={(e) => setScanType(e.target.value)}
              className="w-full bg-black/40 border border-white/10 rounded-xl p-4 text-white focus:ring-1 focus:ring-cyan-500 outline-none cursor-pointer"
            >
              <option value="xray">Radiography (X-Ray)</option>
              <option value="ct">Computed Tomography (CT)</option>
            </select>

            <motion.button
              whileHover={!loading && file ? { scale: 1.02 } : {}} whileTap={!loading && file ? { scale: 0.98 } : {}}
              onClick={handleAnalyze} disabled={!file || loading}
              className="w-full bg-cyan-600 hover:bg-cyan-500 disabled:bg-white/5 disabled:text-slate-500 disabled:border disabled:border-white/5 text-white font-semibold py-4 rounded-xl flex items-center justify-center gap-3 shadow-[0_0_20px_rgba(6,182,212,0.2)] transition-all"
            >
              {loading ? <Loader2 className="animate-spin w-5 h-5" /> : <Fingerprint className="w-5 h-5" />}
              {loading ? 'Processing Tensors...' : 'Run Diagnostics'}
            </motion.button>
          </div>
        </div>

        <div 
          onClick={() => fileInputRef.current.click()}
          className="w-full md:w-2/3 h-80 border border-dashed border-white/20 hover:border-cyan-500/50 rounded-2xl flex flex-col items-center justify-center cursor-pointer bg-black/20 relative overflow-hidden group transition-all"
        >
          <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept="image/*" />
          {preview ? (
            <motion.img initial={{ opacity: 0 }} animate={{ opacity: 1 }} src={preview} className="h-full w-auto object-contain z-10 p-4" />
          ) : (
            <div className="flex flex-col items-center text-slate-500 group-hover:text-cyan-400 transition-colors">
              <UploadCloud className="w-16 h-16 mb-4 opacity-50 group-hover:opacity-100" />
              <p className="font-medium text-lg">Select DICOM/Image file</p>
              <p className="text-xs mt-2 opacity-60">End-to-end encrypted processing</p>
            </div>
          )}
        </div>
      </motion.div>

      <AnimatePresence>
        {results && (
          <motion.div 
            initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, scale: 0.95 }}
            className="glass-panel rounded-3xl p-8"
          >
            <div className="flex justify-between items-end mb-8 border-b border-white/10 pb-6">
              <div>
                <h3 className="text-2xl font-bold text-white">XAI Matrix</h3>
                <p className="text-slate-400 text-sm mt-1">Class activation mapping across independent architectures</p>
              </div>
              <button onClick={() => setResults(null)} className="text-slate-400 hover:text-white flex items-center gap-2 bg-white/5 hover:bg-white/10 px-4 py-2 rounded-lg transition-colors border border-white/5">
                <RefreshCw className="w-4 h-4" /> Clear Cache
              </button>
            </div>

            <div className="space-y-6">
              {['DenseNet169', 'ResNet50', 'Swin Transformer'].map((modelName, i) => {
                const data = results[modelName.split(' ')[0].toLowerCase()];
                return (
                  <motion.div 
                    key={modelName}
                    initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.15 }}
                    className="grid grid-cols-1 lg:grid-cols-4 gap-6 items-center bg-black/40 p-6 rounded-2xl border border-white/5 hover:border-white/10 transition-colors"
                  >
                    <div className="space-y-1 pl-2">
                      <h4 className="font-bold text-lg text-white">{modelName}</h4>
                      <p className="text-xs text-cyan-400/70 font-mono">confidence bounds resolved</p>
                    </div>
                    
                    {[
                      { title: "Source", img: results.original },
                      { title: "Heatmap", img: data.heatmap },
                      { title: "Composite", img: data.overlay, highlight: true }
                    ].map((col, idx) => (
                      <div key={idx} className="flex flex-col">
                        <p className="text-xs uppercase tracking-widest text-slate-500 mb-3 pl-1">{col.title}</p>
                        <motion.div whileHover={{ scale: 1.03 }} className={`relative rounded-xl overflow-hidden aspect-square ${col.highlight ? 'ring-2 ring-cyan-500/50 shadow-[0_0_15px_rgba(6,182,212,0.15)]' : 'border border-white/10'}`}>
                          <img src={col.img} className="w-full h-full object-cover" alt={col.title} />
                        </motion.div>
                      </div>
                    ))}
                  </motion.div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}