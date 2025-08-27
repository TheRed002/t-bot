/**
 * Visual Strategy Canvas - Node-based strategy builder
 */

import React from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Brain } from 'lucide-react';

interface VisualStrategyCanvasProps {
  nodes: any[];
  connections: any[];
  zoom: number;
  offset: { x: number; y: number };
  showGrid: boolean;
  dataFlowAnimation: boolean;
  selectedNodes: string[];
  onNodeAdd: (nodeType: string, position: { x: number; y: number }) => void;
  onNodeUpdate: (nodeId: string, updates: any) => void;
  onNodeDelete: (nodeId: string) => void;
  onNodeSelect: (nodeIds: string[]) => void;
  onNodeConnect: (fromNode: string, toNode: string, connectionType: string) => void;
  onCanvasClick: () => void;
}

const VisualStrategyCanvas: React.FC<VisualStrategyCanvasProps> = ({
  nodes,
  connections,
  zoom,
  offset,
  showGrid,
  dataFlowAnimation,
  selectedNodes,
  onNodeAdd,
  onNodeUpdate,
  onNodeDelete,
  onNodeSelect,
  onNodeConnect,
  onCanvasClick
}) => {
  return (
    <div className="w-full h-full relative bg-gradient-to-br from-gray-900 to-gray-800 overflow-hidden">
      {showGrid && (
        <div 
          className="absolute inset-0 opacity-20"
          style={{
            backgroundImage: 'radial-gradient(circle, #333 1px, transparent 1px)',
            backgroundSize: '20px 20px',
            transform: `scale(${zoom}) translate(${offset.x}px, ${offset.y}px)`
          }}
        />
      )}
      
      <div 
        className="absolute inset-0 cursor-pointer"
        onClick={onCanvasClick}
        style={{ transform: `scale(${zoom}) translate(${offset.x}px, ${offset.y}px)` }}
      >
        {/* Render nodes */}
        {nodes.map((node) => (
          <div
            key={node.id}
            className="absolute transform -translate-x-1/2 -translate-y-1/2"
            style={{
              left: node.position.x,
              top: node.position.y
            }}
            onClick={(e) => {
              e.stopPropagation();
              onNodeSelect([node.id]);
            }}
          >
            <Card 
              className={`w-32 h-20 cursor-pointer border-2 transition-all ${
                selectedNodes.includes(node.id) 
                  ? 'border-primary bg-primary/10' 
                  : 'border-gray-600 hover:border-gray-400'
              }`}
            >
              <CardContent className="p-2 flex flex-col items-center justify-center text-center">
                <Brain className="h-6 w-6 mb-1" />
                <div className="text-xs font-medium truncate w-full">
                  {node.type}
                </div>
                <div className="text-xs text-muted-foreground">
                  {node.status || 'idle'}
                </div>
              </CardContent>
            </Card>
          </div>
        ))}

        {/* Render connections */}
        <svg className="absolute inset-0 pointer-events-none">
          {connections.map((connection) => {
            const fromNode = nodes.find(n => n.id === connection.from);
            const toNode = nodes.find(n => n.id === connection.to);
            
            if (!fromNode || !toNode) return null;
            
            return (
              <line
                key={connection.id}
                x1={fromNode.position.x}
                y1={fromNode.position.y}
                x2={toNode.position.x}
                y2={toNode.position.y}
                stroke={dataFlowAnimation ? '#00ff00' : '#666'}
                strokeWidth="2"
                className={dataFlowAnimation ? 'animate-pulse' : ''}
              />
            );
          })}
        </svg>
      </div>

      {nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center text-muted-foreground">
            <Brain className="h-16 w-16 mx-auto mb-4 opacity-50" />
            <h3 className="text-lg font-medium mb-2">Build Your Strategy</h3>
            <p className="text-sm">
              Drag components from the palette to create your trading strategy
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default VisualStrategyCanvas;