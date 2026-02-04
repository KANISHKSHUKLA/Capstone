import React, { useState, useRef, useEffect } from 'react';
import { FiSend, FiX, FiMessageSquare } from 'react-icons/fi';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ChatMessage {
  timestamp: string;
  query: string;
  response: string;
}

interface ChatPanelProps {
  jobId: string;
}

const ChatPanel: React.FC<ChatPanelProps> = ({ jobId }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [panelWidth, setPanelWidth] = useState(384);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const resizeRef = useRef<HTMLDivElement>(null);
  const [isResizing, setIsResizing] = useState(false);
  const minWidth = 320; 
  const maxWidth = 1200; 

  useEffect(() => {
    if (isOpen) {
      fetchChatHistory();
    }
  }, [isOpen, jobId]);

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory]);

  const fetchChatHistory = async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/jobs/${jobId}/chat`);
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      const data = await response.json();
      setChatHistory(data.chat_history || []);
    } catch (error) {
      console.error('Error fetching chat history:', error);
    }
  };

  const sendMessage = async () => {
    if (!message.trim()) return;
    
    const currentMessage = message;
    setMessage('');
    setIsLoading(true);
    
    const tempTimestamp = new Date().toISOString();
    const tempChat = {
      timestamp: tempTimestamp,
      query: currentMessage,
      response: '' 
    };
    setChatHistory(prev => [...prev, tempChat]);
    
    try {
      const response = await fetch(`http://localhost:8000/api/jobs/${jobId}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: currentMessage }),
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      await response.json();
      fetchChatHistory();
      
    } catch (error) {
      console.error('Error sending message:', error);
      setChatHistory(prev => {
        const updated = [...prev];
        const lastIndex = updated.length - 1;
        if (lastIndex >= 0) {
          updated[lastIndex] = {
            ...updated[lastIndex],
            response: 'Sorry, there was an error processing your request. Please try again.'
          };
        }
        return updated;
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;
      
      const newWidth = window.innerWidth - e.clientX;
      
      if (newWidth >= minWidth && newWidth <= maxWidth) {
        setPanelWidth(newWidth);
      }
    };
    
    const handleMouseUp = () => {
      setIsResizing(false);
    };
    
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }
    
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing]);
  
  const startResizing = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  };
  
  const togglePanel = () => {
    setIsOpen(!isOpen);
  };
  
  const renderMessageContent = (content: string) => {
    return (
      <div className="markdown-content">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            code({ node, className, children, ...props }: any) {
              const match = /language-(\w+)/.exec(className || '');
              const isCodeBlock = !props.inline && match;
              return isCodeBlock ? (
                <div className="my-2 rounded-md overflow-hidden">
                  <SyntaxHighlighter
                    language={match ? match[1] : ''}
                    style={tomorrow}
                    customStyle={{ margin: 0 }}
                    wrapLines={true}
                    wrapLongLines={true}
                  >
                    {String(children).replace(/\n$/, '')}
                  </SyntaxHighlighter>
                </div>
              ) : (
                <code className={className} {...props}>
                  {children}
                </code>
              );
            },
            p: ({ children }: any) => <p className="whitespace-pre-wrap mb-2">{children}</p>,
            a: ({ href, children }: any) => (
              <a 
                href={href} 
                target="_blank" 
                rel="noopener noreferrer" 
                className="text-primary-600 hover:underline"
              >
                {children}
              </a>
            ),
            ul: ({ children }: any) => <ul className="list-disc pl-5 mb-4">{children}</ul>,
            ol: ({ children }: any) => <ol className="list-decimal pl-5 mb-4">{children}</ol>,
            li: ({ children }: any) => <li className="mb-1">{children}</li>,
            h1: ({ children }: any) => <h1 className="text-2xl font-bold my-4">{children}</h1>,
            h2: ({ children }: any) => <h2 className="text-xl font-bold my-3">{children}</h2>,
            h3: ({ children }: any) => <h3 className="text-lg font-bold my-2">{children}</h3>,
            h4: ({ children }: any) => <h4 className="text-base font-bold my-1">{children}</h4>,
            blockquote: ({ children }: any) => (
              <blockquote className="border-l-4 border-neutral-300 pl-4 italic my-2">
                {children}
              </blockquote>
            ),
            hr: () => <hr className="my-4 border-neutral-300" />,
            table: ({ children }: any) => (
              <div className="overflow-x-auto my-4">
                <table className="border-collapse border border-neutral-300 w-full">
                  {children}
                </table>
              </div>
            ),
            thead: ({ children }: any) => <thead className="bg-neutral-100">{children}</thead>,
            tbody: ({ children }: any) => <tbody>{children}</tbody>,
            tr: ({ children }: any) => <tr>{children}</tr>,
            th: ({ children }: any) => (
              <th className="border border-neutral-300 px-4 py-2 text-left font-bold">{children}</th>
            ),
            td: ({ children }: any) => <td className="border border-neutral-300 px-4 py-2">{children}</td>,
          }}
        >
          {content}
        </ReactMarkdown>
      </div>
    );
  };

  return (
    <>
      {/* Chat Toggle Button */}
      <button 
        onClick={togglePanel}
        className="fixed bottom-6 right-6 z-20 bg-primary-600 text-white p-3 rounded-full shadow-lg hover:bg-primary-700 transition-colors"
        aria-label={isOpen ? "Close chat" : "Open chat"}
      >
        {isOpen ? <FiX size={20} /> : <FiMessageSquare size={20} />}
      </button>

      {/* Resize Handle */}
      {isOpen && (
        <div
          ref={resizeRef}
          className="fixed z-20 w-1 h-full bg-primary-200 hover:bg-primary-400 cursor-col-resize"
          style={{ 
            left: `calc(100% - ${panelWidth}px - 1px)`,
            top: 0
          }}
          onMouseDown={startResizing}
        />
      )}

      {/* Chat Panel */}
      <div 
        className="fixed right-0 top-0 h-full bg-white shadow-lg z-10 transition-all duration-300 ease-in-out overflow-hidden flex flex-col"
        style={{ 
          width: isOpen ? `${panelWidth}px` : '0',
        }}
      >
        <div className="p-4 border-b border-neutral-200 bg-primary-50 flex justify-between items-center">
          <h2 className="text-lg font-semibold text-primary-700">Paper Chat</h2>
          <button 
            onClick={togglePanel}
            className="text-neutral-500 hover:text-neutral-700"
            aria-label="Close chat"
          >
            <FiX size={20} />
          </button>
        </div>

        {/* Chat Messages */}
        <div className="flex-grow overflow-y-auto p-4 bg-neutral-50">
          {chatHistory.length === 0 ? (
            <div className="text-center text-neutral-500 mt-8">
              <p>No messages yet. Ask a question about the paper!</p>
            </div>
          ) : (
            chatHistory.map((chat, index) => (
              <div key={index} className="mb-4">
                {/* User message */}
                <div className="flex justify-end mb-2">
                  <div className="bg-primary-100 text-primary-800 p-3 rounded-lg max-w-[80%] break-words">
                    {chat.query}
                  </div>
                </div>
                
                {/* AI response */}
                <div className="flex justify-start">
                  <div className="bg-white border border-neutral-200 p-3 rounded-lg shadow-sm max-w-[80%] break-words">
                    {renderMessageContent(chat.response)}
                  </div>
                </div>
              </div>
            ))
          )}
          {/* Loading animation */}
          {isLoading && (
            <div className="flex justify-start my-4">
              <div className="bg-white border border-neutral-200 p-3 rounded-lg shadow-sm break-words">
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                  <div className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style={{ animationDelay: '600ms' }}></div>
                </div>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Message Input */}
        <div className="p-4 border-t border-neutral-200 bg-white">
          <div className="flex items-center">
            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about the paper..."
              className="flex-grow p-2 border border-neutral-300 rounded-l-md focus:outline-none focus:ring-2 focus:ring-primary-500 resize-none"
              rows={2}
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              disabled={isLoading || !message.trim()}
              className={`p-3 rounded-r-md ${
                isLoading || !message.trim()
                  ? 'bg-neutral-300 text-neutral-500'
                  : 'bg-primary-600 text-white hover:bg-primary-700'
              } transition-colors`}
              aria-label="Send message"
            >
              <FiSend size={18} />
            </button>
          </div>
        </div>
      </div>
    </>
  );
};

export default ChatPanel;
