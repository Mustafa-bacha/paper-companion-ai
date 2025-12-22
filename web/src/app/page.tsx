"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Brain,
  FileText,
  MessageSquare,
  Sparkles,
  ChevronRight,
  Cpu,
  Database,
  Zap,
  BookOpen,
  Search,
  Loader2,
  Check,
  AlertCircle,
} from "lucide-react";

import Phase1Playground from "@/components/Phase1Playground";
import Phase2RAG from "@/components/Phase2RAG";
import Header from "@/components/Header";

type TabType = "home" | "phase1" | "phase2";

interface HealthStatus {
  status: string;
  phase1_loaded: boolean;
  phase2_loaded: boolean;
  device: string;
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabType>("home");
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      }
    } catch (error) {
      console.error("API not available:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen">
      <Header activeTab={activeTab} setActiveTab={setActiveTab} health={health} />
      
      <main className="container mx-auto px-4 py-8">
        <AnimatePresence mode="wait">
          {activeTab === "home" && (
            <motion.div
              key="home"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <HomePage 
                setActiveTab={setActiveTab} 
                health={health} 
                loading={loading} 
              />
            </motion.div>
          )}
          
          {activeTab === "phase1" && (
            <motion.div
              key="phase1"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
            >
              <Phase1Playground />
            </motion.div>
          )}
          
          {activeTab === "phase2" && (
            <motion.div
              key="phase2"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
            >
              <Phase2RAG />
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

function HomePage({ 
  setActiveTab, 
  health, 
  loading 
}: { 
  setActiveTab: (tab: TabType) => void;
  health: HealthStatus | null;
  loading: boolean;
}) {
  return (
    <div className="max-w-6xl mx-auto space-y-12">
      {/* Hero Section */}
      <section className="text-center py-12">
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-500/10 rounded-full text-blue-400 text-sm mb-6">
            <Sparkles className="w-4 h-4" />
            AI-Powered Research Assistant
          </div>
          
          <h1 className="text-5xl md:text-6xl font-bold text-white mb-6">
            Research Paper
            <span className="bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
              {" "}Companion AI
            </span>
          </h1>
          
          <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-8">
            Explore custom transformer architectures built from scratch and interact with 
            a powerful RAG system trained on 500+ research papers.
          </p>

          {/* Status Badge */}
          <div className="inline-flex items-center gap-3 px-4 py-2 bg-gray-800 rounded-lg">
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
            ) : health?.status === "healthy" ? (
              <Check className="w-4 h-4 text-green-400" />
            ) : (
              <AlertCircle className="w-4 h-4 text-yellow-400" />
            )}
            <span className="text-gray-300 text-sm">
              {loading ? "Connecting to API..." : 
               health?.status === "healthy" ? `Running on ${health.device.toUpperCase()}` :
               "API Offline - Start the backend server"}
            </span>
          </div>
        </motion.div>
      </section>

      {/* Feature Cards */}
      <section className="grid md:grid-cols-2 gap-6">
        {/* Phase 1 Card */}
        <motion.div
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => setActiveTab("phase1")}
          className="cursor-pointer group"
        >
          <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-blue-600/20 to-blue-800/20 border border-blue-500/30 p-8 h-full">
            <div className="absolute top-0 right-0 w-40 h-40 bg-blue-500/10 rounded-full blur-3xl" />
            
            <div className="relative z-10">
              <div className="flex items-center gap-4 mb-6">
                <div className="p-3 bg-blue-500/20 rounded-xl">
                  <Brain className="w-8 h-8 text-blue-400" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-white">Phase 1</h2>
                  <p className="text-blue-400">Transformer Playground</p>
                </div>
              </div>
              
              <p className="text-gray-400 mb-6">
                Interact with custom transformer models built from scratch:
              </p>
              
              <div className="space-y-3">
                <FeatureItem 
                  icon={<Cpu className="w-5 h-5" />} 
                  title="Text Classifier"
                  description="Encoder-only model for paper categorization"
                  status={health?.phase1_loaded}
                />
                <FeatureItem 
                  icon={<FileText className="w-5 h-5" />} 
                  title="Language Model"
                  description="Decoder-only model for text generation"
                  status={health?.phase1_loaded}
                />
                <FeatureItem 
                  icon={<Zap className="w-5 h-5" />} 
                  title="Summarizer"
                  description="Encoder-decoder for TL;DR generation"
                  status={health?.phase1_loaded}
                />
              </div>
              
              <div className="mt-6 flex items-center text-blue-400 group-hover:translate-x-2 transition-transform">
                <span>Explore Phase 1</span>
                <ChevronRight className="w-5 h-5 ml-1" />
              </div>
            </div>
          </div>
        </motion.div>

        {/* Phase 2 Card */}
        <motion.div
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => setActiveTab("phase2")}
          className="cursor-pointer group"
        >
          <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-purple-600/20 to-purple-800/20 border border-purple-500/30 p-8 h-full">
            <div className="absolute top-0 right-0 w-40 h-40 bg-purple-500/10 rounded-full blur-3xl" />
            
            <div className="relative z-10">
              <div className="flex items-center gap-4 mb-6">
                <div className="p-3 bg-purple-500/20 rounded-xl">
                  <MessageSquare className="w-8 h-8 text-purple-400" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-white">Phase 2</h2>
                  <p className="text-purple-400">RAG Q&A System</p>
                </div>
              </div>
              
              <p className="text-gray-400 mb-6">
                Ask questions about AI research with citation support:
              </p>
              
              <div className="space-y-3">
                <FeatureItem 
                  icon={<Database className="w-5 h-5" />} 
                  title="55,986 Chunks"
                  description="Indexed from 500 research papers"
                  status={health?.phase2_loaded}
                />
                <FeatureItem 
                  icon={<Search className="w-5 h-5" />} 
                  title="Semantic Search"
                  description="FAISS vector store with MiniLM embeddings"
                  status={health?.phase2_loaded}
                />
                <FeatureItem 
                  icon={<BookOpen className="w-5 h-5" />} 
                  title="Citations"
                  description="Every answer includes paper references"
                  status={health?.phase2_loaded}
                />
              </div>
              
              <div className="mt-6 flex items-center text-purple-400 group-hover:translate-x-2 transition-transform">
                <span>Explore Phase 2</span>
                <ChevronRight className="w-5 h-5 ml-1" />
              </div>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Stats Section */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: "Papers Indexed", value: "500+", icon: FileText },
          { label: "Text Chunks", value: "55,986", icon: Database },
          { label: "Model Parameters", value: "~22M", icon: Cpu },
          { label: "Categories", value: "5", icon: Brain },
        ].map((stat, index) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="bg-gray-800/50 border border-gray-700 rounded-xl p-4 text-center"
          >
            <stat.icon className="w-6 h-6 text-gray-400 mx-auto mb-2" />
            <div className="text-2xl font-bold text-white">{stat.value}</div>
            <div className="text-sm text-gray-400">{stat.label}</div>
          </motion.div>
        ))}
      </section>

      {/* Model Architecture Info */}
      <section className="bg-gray-800/30 border border-gray-700 rounded-2xl p-8">
        <h3 className="text-xl font-bold text-white mb-6">Model Architecture</h3>
        <div className="grid md:grid-cols-3 gap-6 text-sm">
          <div>
            <h4 className="text-blue-400 font-semibold mb-2">Encoder (Classifier)</h4>
            <ul className="text-gray-400 space-y-1">
              <li>• d_model: 256</li>
              <li>• Layers: 4</li>
              <li>• Heads: 4</li>
              <li>• Dropout: 0.3</li>
              <li>• 78.1% accuracy</li>
            </ul>
          </div>
          <div>
            <h4 className="text-green-400 font-semibold mb-2">Decoder (LM)</h4>
            <ul className="text-gray-400 space-y-1">
              <li>• d_model: 256</li>
              <li>• Layers: 4</li>
              <li>• Heads: 4</li>
              <li>• Dropout: 0.15</li>
              <li>• 46.12 PPL</li>
            </ul>
          </div>
          <div>
            <h4 className="text-purple-400 font-semibold mb-2">Encoder-Decoder (Seq2Seq)</h4>
            <ul className="text-gray-400 space-y-1">
              <li>• d_model: 256</li>
              <li>• Layers: 4 + 4</li>
              <li>• Heads: 4</li>
              <li>• Dropout: 0.2</li>
              <li>• ROUGE-1: 0.31</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}

function FeatureItem({ 
  icon, 
  title, 
  description, 
  status 
}: { 
  icon: React.ReactNode;
  title: string;
  description: string;
  status?: boolean;
}) {
  return (
    <div className="flex items-start gap-3 bg-white/5 rounded-lg p-3">
      <div className="text-gray-400">{icon}</div>
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <span className="text-white font-medium">{title}</span>
          {status !== undefined && (
            <span className={`w-2 h-2 rounded-full ${status ? 'bg-green-400' : 'bg-gray-500'}`} />
          )}
        </div>
        <span className="text-gray-500 text-sm">{description}</span>
      </div>
    </div>
  );
}
