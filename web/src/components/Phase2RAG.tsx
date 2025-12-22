"use client";

import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  MessageSquare,
  Send,
  Loader2,
  BookOpen,
  FileText,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
  Sparkles,
  Database,
  Search,
  Settings,
  User,
  Bot,
  AlertCircle,
  RefreshCw,
} from "lucide-react";

interface RAGSource {
  paper_id: string;
  section: string;
  page: number;
  text: string;
  score: number;
}

interface RAGResponse {
  answer: string;
  confidence: number;
  sources: RAGSource[];
  model_used: string;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: RAGSource[];
  confidence?: number;
  model?: string;
  timestamp: Date;
}

export default function Phase2RAG() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Settings
  const [selectedModel, setSelectedModel] = useState<"flan-t5-small" | "flan-t5-base">("flan-t5-small");
  const [numSources, setNumSources] = useState(5);
  const [showSettings, setShowSettings] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const res = await fetch("/api/phase2/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: input,
          model: selectedModel,
          num_sources: numSources,
          show_sources: true,
        }),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "API request failed");
      }

      const data: RAGResponse = await res.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.answer,
        sources: data.sources,
        confidence: data.confidence,
        model: data.model_used,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setError(null);
  };

  const sampleQuestions = [
    "What are the main challenges in retrieval augmented generation?",
    "How do transformers handle long sequences?",
    "What methods are used for efficient fine-tuning of LLMs?",
    "What are the recent advances in attention mechanisms?",
    "How do large language models handle hallucinations?",
  ];

  return (
    <div className="max-w-5xl mx-auto">
      {/* Header */}
      <div className="text-center mb-6">
        <h1 className="text-3xl font-bold text-white mb-2">
          Phase 2: RAG Q&A System
        </h1>
        <p className="text-gray-400">
          Ask questions about AI research with citations from 500+ papers
        </p>
      </div>

      {/* Stats Bar */}
      <div className="flex flex-wrap items-center justify-center gap-4 mb-6">
        <StatBadge icon={Database} label="55,986 chunks" />
        <StatBadge icon={FileText} label="500 papers" />
        <StatBadge icon={Search} label="Semantic search" />
        <StatBadge icon={BookOpen} label="With citations" />
      </div>

      {/* Main Chat Interface */}
      <div className="bg-gray-800/50 border border-gray-700 rounded-2xl overflow-hidden">
        {/* Settings Bar */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700 bg-gray-800/50">
          <div className="flex items-center gap-4">
            <ModelSelector 
              value={selectedModel} 
              onChange={setSelectedModel} 
            />
            <div className="hidden md:flex items-center gap-2 text-sm text-gray-400">
              <span>Sources:</span>
              <select
                value={numSources}
                onChange={(e) => setNumSources(Number(e.target.value))}
                className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-white text-sm"
              >
                {[3, 5, 7, 10].map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 hover:bg-gray-700 rounded-lg transition-colors md:hidden"
            >
              <Settings className="w-5 h-5 text-gray-400" />
            </button>
            {messages.length > 0 && (
              <button
                onClick={clearChat}
                className="flex items-center gap-1 px-3 py-1.5 text-sm text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                <span className="hidden md:inline">Clear</span>
              </button>
            )}
          </div>
        </div>

        {/* Mobile Settings Dropdown */}
        <AnimatePresence>
          {showSettings && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="border-b border-gray-700 overflow-hidden md:hidden"
            >
              <div className="p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">Number of sources:</span>
                  <select
                    value={numSources}
                    onChange={(e) => setNumSources(Number(e.target.value))}
                    className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-white text-sm"
                  >
                    {[3, 5, 7, 10].map((n) => (
                      <option key={n} value={n}>{n}</option>
                    ))}
                  </select>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Messages Area */}
        <div className="h-[500px] overflow-y-auto p-4 space-y-4">
          {messages.length === 0 ? (
            <EmptyState 
              onSelectQuestion={(q) => setInput(q)} 
              sampleQuestions={sampleQuestions}
            />
          ) : (
            messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))
          )}
          
          {loading && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-start gap-3"
            >
              <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center">
                <Bot className="w-5 h-5 text-purple-400" />
              </div>
              <div className="flex items-center gap-2 px-4 py-3 bg-gray-700/50 rounded-2xl">
                <Loader2 className="w-4 h-4 animate-spin text-purple-400" />
                <span className="text-gray-400">Searching papers and generating answer...</span>
              </div>
            </motion.div>
          )}
          
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400"
            >
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              <span>{error}</span>
            </motion.div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-700 p-4">
          <div className="flex gap-3">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about AI research..."
              rows={1}
              className="flex-1 bg-gray-700/50 border border-gray-600 rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 resize-none"
              style={{ minHeight: "48px", maxHeight: "120px" }}
            />
            <button
              onClick={handleSubmit}
              disabled={loading || !input.trim()}
              className="px-4 py-3 bg-gradient-to-r from-purple-500 to-blue-600 text-white rounded-xl hover:from-purple-600 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              {loading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>
          <p className="mt-2 text-xs text-gray-500 text-center">
            Powered by {selectedModel} • Press Enter to send
          </p>
        </div>
      </div>
    </div>
  );
}

function StatBadge({ icon: Icon, label }: { icon: React.ElementType; label: string }) {
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-full text-sm text-gray-400">
      <Icon className="w-4 h-4" />
      <span>{label}</span>
    </div>
  );
}

function ModelSelector({ 
  value, 
  onChange 
}: { 
  value: "flan-t5-small" | "flan-t5-base";
  onChange: (v: "flan-t5-small" | "flan-t5-base") => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-sm text-gray-400">Model:</span>
      <div className="flex bg-gray-700 rounded-lg p-0.5">
        <button
          onClick={() => onChange("flan-t5-small")}
          className={`px-3 py-1 text-sm rounded-md transition-colors ${
            value === "flan-t5-small"
              ? "bg-purple-500 text-white"
              : "text-gray-400 hover:text-white"
          }`}
        >
          Small
        </button>
        <button
          onClick={() => onChange("flan-t5-base")}
          className={`px-3 py-1 text-sm rounded-md transition-colors ${
            value === "flan-t5-base"
              ? "bg-purple-500 text-white"
              : "text-gray-400 hover:text-white"
          }`}
        >
          Base
        </button>
      </div>
    </div>
  );
}

function EmptyState({ 
  onSelectQuestion, 
  sampleQuestions 
}: { 
  onSelectQuestion: (q: string) => void;
  sampleQuestions: string[];
}) {
  return (
    <div className="h-full flex flex-col items-center justify-center text-center px-4">
      <div className="w-16 h-16 rounded-full bg-purple-500/20 flex items-center justify-center mb-4">
        <MessageSquare className="w-8 h-8 text-purple-400" />
      </div>
      <h3 className="text-xl font-semibold text-white mb-2">
        Ask about AI Research
      </h3>
      <p className="text-gray-400 mb-6 max-w-md">
        Get answers from 500+ indexed research papers with proper citations
      </p>
      
      <div className="w-full max-w-lg space-y-2">
        <p className="text-sm text-gray-500 mb-2">Try asking:</p>
        {sampleQuestions.slice(0, 4).map((q, idx) => (
          <button
            key={idx}
            onClick={() => onSelectQuestion(q)}
            className="w-full text-left px-4 py-3 bg-gray-700/30 hover:bg-gray-700/50 border border-gray-700 hover:border-purple-500/50 rounded-xl text-gray-300 text-sm transition-all"
          >
            <Sparkles className="w-4 h-4 inline-block mr-2 text-purple-400" />
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const [showSources, setShowSources] = useState(false);
  const [copied, setCopied] = useState(false);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (message.role === "user") {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-start gap-3 justify-end"
      >
        <div className="max-w-[80%] px-4 py-3 bg-blue-600 rounded-2xl rounded-tr-sm">
          <p className="text-white">{message.content}</p>
        </div>
        <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center flex-shrink-0">
          <User className="w-5 h-5 text-blue-400" />
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex items-start gap-3"
    >
      <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center flex-shrink-0">
        <Bot className="w-5 h-5 text-purple-400" />
      </div>
      
      <div className="flex-1 max-w-[85%] space-y-2">
        {/* Answer */}
        <div className="px-4 py-3 bg-gray-700/50 rounded-2xl rounded-tl-sm">
          <p className="text-gray-200 leading-relaxed">{message.content}</p>
        </div>

        {/* Meta Info */}
        <div className="flex items-center gap-3 text-xs text-gray-500">
          {message.confidence !== undefined && (
            <span className="flex items-center gap-1">
              <div 
                className={`w-2 h-2 rounded-full ${
                  message.confidence > 0.7 ? 'bg-green-400' :
                  message.confidence > 0.4 ? 'bg-yellow-400' : 'bg-red-400'
                }`}
              />
              {(message.confidence * 100).toFixed(0)}% confidence
            </span>
          )}
          {message.model && (
            <span>{message.model}</span>
          )}
          <button
            onClick={copyToClipboard}
            className="p-1 hover:bg-gray-700 rounded transition-colors"
          >
            {copied ? (
              <Check className="w-3 h-3 text-green-400" />
            ) : (
              <Copy className="w-3 h-3" />
            )}
          </button>
        </div>

        {/* Sources Toggle */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-2">
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-2 text-sm text-purple-400 hover:text-purple-300 transition-colors"
            >
              <BookOpen className="w-4 h-4" />
              <span>{message.sources.length} sources</span>
              {showSources ? (
                <ChevronUp className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
            </button>

            <AnimatePresence>
              {showSources && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="mt-2 space-y-2 overflow-hidden"
                >
                  {message.sources.map((source, idx) => (
                    <SourceCard key={idx} source={source} index={idx + 1} />
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}
      </div>
    </motion.div>
  );
}

function SourceCard({ source, index }: { source: RAGSource; index: number }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-3 text-sm">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="w-5 h-5 flex items-center justify-center bg-purple-500/20 rounded text-purple-400 text-xs font-medium">
            {index}
          </span>
          <span className="text-white font-medium">{source.paper_id}</span>
          <span className="text-gray-500">• {source.section}</span>
        </div>
        <span className="text-gray-500 text-xs">
          Score: {source.score.toFixed(2)}
        </span>
      </div>
      
      <p className={`mt-2 text-gray-400 ${!expanded ? 'line-clamp-2' : ''}`}>
        {source.text}
      </p>
      
      {source.text.length > 150 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-1 text-purple-400 text-xs hover:text-purple-300"
        >
          {expanded ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  );
}
