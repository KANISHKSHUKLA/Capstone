import React, { useRef, useEffect, useState, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
// import ForceGraph3D from 'react-force-graph-3d';
import * as d3 from 'd3';
import { ResizableBox } from 'react-resizable';

interface Node {
  id: string;
  label: string;
  type: string;
  description: string;
  // Added dynamically
  color?: string;
  val?: number;
}

interface Edge {
  source: string;
  target: string;
  label: string;
  description: string;
}

interface GraphData {
  nodes: Node[];
  edges: Edge[];
  metadata?: {
    paper_title?: string;
    node_count?: number;
    edge_count?: number;
  };
}

interface KnowledgeGraphProps {
  graphData: GraphData;
  width?: number;
  height?: number;
  resizable?: boolean;
}

const NODE_COLORS = {
  concept: '#4285F4',    // Blue
  method: '#EA4335',     // Red
  dataset: '#FBBC05',    // Yellow
  result: '#34A853',     // Green
  entity: '#9C27B0',     // Purple
  limitation: '#FF5722', // Deep Orange
  application: '#00BCD4',// Cyan
  default: '#757575'     // Gray
};

const NODE_SIZES = {
  concept: 8,
  method: 7,
  dataset: 6,
  result: 6,
  entity: 5,
  limitation: 5,
  application: 7,
  default: 4
};

const KnowledgeGraph: React.FC<KnowledgeGraphProps> = ({ 
  graphData, 
  width = 800, 
  height = 600,
  resizable = true 
}) => {
  const [use3D, setUse3D] = useState<boolean>(false);
  const [hoveredNode, setHoveredNode] = useState<Node | null>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [transformedData, setTransformedData] = useState<{nodes: any[], links: any[]}>({ nodes: [], links: [] });
  const [loading, setLoading] = useState<boolean>(true);
  const [mousePosition, setMousePosition] = useState<{x: number, y: number}>({x: 0, y: 0});
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [filters, setFilters] = useState<{[key: string]: boolean}>({
    concept: true,
    method: true,
    dataset: true,
    result: true,
    entity: true
  });
  
  const graphRef = useRef<any>();
  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      setMousePosition({ x: event.clientX, y: event.clientY });
    };
    
    window.addEventListener('mousemove', handleMouseMove);
    
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);
  
  useEffect(() => {
    if (!graphData || !graphData.nodes || !graphData.edges) {
      setTransformedData({ nodes: [], links: [] });
      setLoading(false);
      return;
    }
    
    try {
      // Process nodes
      const nodes = graphData.nodes.map(node => ({
        ...node,
        color: NODE_COLORS[node.type as keyof typeof NODE_COLORS] || NODE_COLORS.default,
        val: NODE_SIZES[node.type as keyof typeof NODE_SIZES] || NODE_SIZES.default
      }));
      
      // Process links (edges)
      const links = graphData.edges.map(edge => ({
        source: edge.source,
        target: edge.target,
        label: edge.label,
        description: edge.description
      }));
      
      setTransformedData({ nodes, links });
      setLoading(false);
    } catch (error) {
      console.error('Error transforming graph data:', error);
      setTransformedData({ nodes: [], links: [] });
      setLoading(false);
    }
  }, [graphData]);
  
  const filteredData = React.useMemo(() => {
    const { nodes, links } = transformedData;
    
    // Filter nodes by type and search term
    const filteredNodes = nodes.filter(node => 
      filters[node.type] && 
      (searchTerm === '' || 
       node.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
       node.description?.toLowerCase().includes(searchTerm.toLowerCase()))
    );
    
    const nodeIds = new Set(filteredNodes.map(n => n.id));
    
    // Keep only links where both source and target are in filtered nodes
    const filteredLinks = links.filter(link => {
      const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
      const targetId = typeof link.target === 'object' ? link.target.id : link.target;
      return nodeIds.has(sourceId) && nodeIds.has(targetId);
    });
    
    return { nodes: filteredNodes, links: filteredLinks };
  }, [transformedData, filters, searchTerm]);
  
  // Handle node click
  const handleNodeClick = useCallback((node: any) => {
    // Only update selected node state without moving the graph
    setSelectedNode(node === selectedNode ? null : node);
    
    // No camera movement or repositioning - this keeps the graph completely static
  }, [selectedNode]);
  
  // Handle node hover
  const handleNodeHover = useCallback((node: any) => {
    setHoveredNode(node || null);
    
    // Set cursor style
    document.body.style.cursor = node ? 'pointer' : 'default';
  }, []);
  
  // Toggle 2D/3D view
  const toggleView = () => {
    setUse3D(!use3D);
    setSelectedNode(null);
  };
  
  // Toggle node type filter
  const toggleFilter = (type: string) => {
    setFilters(prev => ({
      ...prev,
      [type]: !prev[type]
    }));
  };
  
  // Helper to get node type color
  const getNodeTypeColor = (type: string) => {
    return NODE_COLORS[type as keyof typeof NODE_COLORS] || NODE_COLORS.default;
  };
  
  // Reset filters
  const resetFilters = () => {
    setFilters({
      concept: true,
      method: true,
      dataset: true,
      result: true,
      entity: true
    });
    setSearchTerm('');
  };
  
  // Generate a UI with graph
  const graphComponent = (
    <div className="relative w-full h-full bg-white rounded-lg border overflow-hidden">
      {loading ? (
        <div className="flex items-center justify-center h-full">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-600"></div>
        </div>
      ) : transformedData.nodes.length === 0 ? (
        <div className="flex items-center justify-center h-full">
          <div className="text-center p-4">
            <h3 className="text-lg font-semibold text-gray-700">No knowledge graph available</h3>
            <p className="text-gray-500 mt-2">The analysis did not generate a knowledge graph for this paper.</p>
          </div>
        </div>
      ) : (
        <>
          {/* Controls */}
          <div className="absolute top-4 right-4 z-10 flex space-x-2">
            <button
              onClick={toggleView}
              className="px-3 py-1 text-sm bg-white border rounded-md shadow-sm hover:bg-gray-50"
            >
              {use3D ? 'Switch to 2D' : 'Switch to 3D'}
            </button>
            <button
              onClick={resetFilters}
              className="px-3 py-1 text-sm bg-white border rounded-md shadow-sm hover:bg-gray-50"
            >
              Reset Filters
            </button>
          </div>
          
          {/* Search */}
          <div className="absolute top-4 left-4 z-10 w-64">
            <input
              type="text"
              placeholder="Search nodes..."
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              className="w-full px-3 py-2 text-sm border rounded-md shadow-sm"
            />
          </div>
          
          {/* Filters */}
          <div className="absolute top-16 left-4 z-10 bg-white p-2 rounded-md shadow-sm border">
            <h4 className="text-xs font-semibold mb-1">Filter by type:</h4>
            {Object.keys(filters).map(type => (
              <div key={type} className="flex items-center text-xs mb-1">
                <input
                  type="checkbox"
                  id={`filter-${type}`}
                  checked={filters[type]}
                  onChange={() => toggleFilter(type)}
                  className="mr-1"
                />
                <span 
                  className="w-3 h-3 inline-block rounded-full mr-1"
                  style={{ backgroundColor: getNodeTypeColor(type) }}
                ></span>
                <label htmlFor={`filter-${type}`} className="capitalize">{type}</label>
              </div>
            ))}
          </div>
          
          {/* Node details */}
          {selectedNode && (
            <div className="absolute bottom-4 left-4 z-10 bg-white p-3 rounded-md shadow-md border w-72 max-h-64 overflow-y-auto">
              <button 
                className="absolute top-1 right-1 text-gray-400 hover:text-gray-600" 
                onClick={() => setSelectedNode(null)}
              >
                ✕
              </button>
              <h3 className="font-bold text-lg mb-1">{selectedNode.label}</h3>
              <div className="text-xs text-gray-500 mb-2 flex items-center">
                <span 
                  className="w-3 h-3 inline-block rounded-full mr-1"
                  style={{ backgroundColor: selectedNode.color }}
                ></span>
                <span className="capitalize">{selectedNode.type}</span>
              </div>
              <p className="text-sm text-gray-700">{selectedNode.description}</p>
              
              {/* Connected nodes */}
              <div className="mt-3 pt-2 border-t border-gray-200">
                <h4 className="text-xs font-semibold mb-1">Connections:</h4>
                <div className="space-y-1">
                  {filteredData.links
                    .filter(link => {
                      const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
                      const targetId = typeof link.target === 'object' ? link.target.id : link.target;
                      return sourceId === selectedNode.id || targetId === selectedNode.id;
                    })
                    .map((link, index) => {
                      const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
                      const targetId = typeof link.target === 'object' ? link.target.id : link.target;
                      const isSource = sourceId === selectedNode.id;
                      
                      const connectedNode = filteredData.nodes.find(n => 
                        n.id === (isSource ? targetId : sourceId)
                      );
                      
                      if (!connectedNode) return null;
                      
                      return (
                        <div key={index} className="text-xs flex items-center">
                          <span className="text-gray-500">
                            {isSource ? 'To' : 'From'} <span className="font-medium">{connectedNode.label}</span> 
                            {link.label && <span> ({link.label})</span>}
                          </span>
                        </div>
                      );
                    })}
                </div>
              </div>
            </div>
          )}
          
          {/* Hover tooltip */}
          {hoveredNode && hoveredNode !== selectedNode && (
          <div 
            className="fixed z-10 bg-white p-2 rounded-md shadow-md border text-xs"
            style={{ 
              left: `${mousePosition.x + 5}px`, 
              top: `${mousePosition.y + 5}px`,
              maxWidth: '200px',
              pointerEvents: 'none' 
            }}
          >
            <div className="font-bold">{hoveredNode.label}</div>
            <div className="text-gray-500 capitalize">{hoveredNode.type}</div>
          </div>
        )}
          
          {/* Graph */}
          <div className="w-full" style={{ height: '600px', position: 'relative', overflow: 'hidden', border: '1px solid #e5e7eb', borderRadius: '0.5rem' }}>
            {use3D ? (
              // 3D graph temporarily disabled
              <div className="flex items-center justify-center h-full bg-gray-100 rounded-lg">
                <div className="text-center p-4">
                  <p className="text-lg font-medium">3D view is temporarily unavailable</p>
                  <p className="text-sm text-gray-600 mt-2">Please use the 2D view instead</p>
                  <button
                    onClick={() => setUse3D(false)}
                    className="mt-4 px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 transition-colors"
                  >
                    Switch to 2D View
                  </button>
                </div>
              </div>
            ) : (
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
                <ForceGraph2D
                  ref={graphRef}
                  graphData={filteredData}
                  nodeLabel="label"
                  nodeColor="color"
                  nodeVal="val"
                  linkColor={() => "#aaaaaa"}
                  linkWidth={0.5}
                  linkDirectionalParticles={2}
                  linkDirectionalParticleWidth={1.2}
                  onNodeClick={handleNodeClick}
                  onNodeHover={handleNodeHover}
                  linkCurvature={0.1}
                  cooldownTicks={100}
                  cooldownTime={2000}
                />
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
  
  // Wrap with ResizableBox if resizable
  if (resizable) {
    return (
      <ResizableBox 
        width={width} 
        height={height}
        minConstraints={[300, 300]}
        maxConstraints={[2000, 1200]}
        className="border rounded-lg shadow-md bg-white overflow-hidden"
      >
        {graphComponent}
      </ResizableBox>
    );
  }
  
  return (
    <div style={{ width, height }}>
      {graphComponent}
    </div>
  );
};

export default KnowledgeGraph;
