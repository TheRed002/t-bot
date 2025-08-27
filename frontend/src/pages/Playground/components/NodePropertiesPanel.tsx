/**
 * Node Properties Panel - Configure selected nodes and view system metrics
 */

import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Alert } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';

import {
  ChevronDown,
  Settings,
  Eye,
  Bug,
  Gauge,
  HardDrive,
  Wifi,
  Activity
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface Node {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: any;
  status?: 'idle' | 'running' | 'error' | 'completed';
}

interface SystemMetrics {
  dataFlowRate: number;
  latency: number;
  cpuUsage: number;
  memoryUsage: number;
}

interface NodePropertiesPanelProps {
  selectedNodes: Node[];
  onNodeUpdate: (nodeId: string, updates: any) => void;
  systemMetrics: SystemMetrics;
}

const NodePropertiesPanel: React.FC<NodePropertiesPanelProps> = ({
  selectedNodes,
  onNodeUpdate,
  systemMetrics
}) => {
  const [activeTab, setActiveTab] = useState('properties');
  const [propertiesExpanded, setPropertiesExpanded] = useState(true);
  const [systemExpanded, setSystemExpanded] = useState(false);

  const selectedNode = selectedNodes[0];

  const handleParameterChange = (parameter: string, value: any) => {
    if (selectedNode) {
      onNodeUpdate(selectedNode.id, {
        data: {
          ...selectedNode.data,
          [parameter]: value
        }
      });
    }
  };

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case 'running': return 'default';
      case 'completed': return 'default';
      case 'error': return 'destructive';
      default: return 'secondary';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'text-blue-500';
      case 'completed': return 'text-green-500';
      case 'error': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  return (
    <div className="h-full flex flex-col bg-gray-900 text-white border-l border-gray-700">
      <div className="p-4 border-b border-gray-700">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Settings className="h-5 w-5" />
          Properties
        </h3>
      </div>

      <div className="flex-1 overflow-auto">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="properties" className="text-xs">Properties</TabsTrigger>
            <TabsTrigger value="logs" className="text-xs">Logs</TabsTrigger>
            <TabsTrigger value="system" className="text-xs">System</TabsTrigger>
          </TabsList>

          <TabsContent value="properties" className="p-4 space-y-4">
            {selectedNode ? (
              <div className="space-y-4">
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base flex items-center justify-between">
                      <span>{selectedNode.type}</span>
                      <Badge variant={getStatusBadgeVariant(selectedNode.status || 'idle')}>
                        {selectedNode.status || 'idle'}
                      </Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <Label htmlFor="node-name" className="text-xs">Node Name</Label>
                      <Input
                        id="node-name"
                        value={selectedNode.data.name || selectedNode.id}
                        onChange={(e) => handleParameterChange('name', e.target.value)}
                        className="mt-1"
                      />
                    </div>

                    <div>
                      <Label className="text-xs">Status</Label>
                      <Select
                        value={selectedNode.status || 'idle'}
                        onValueChange={(value) => onNodeUpdate(selectedNode.id, { status: value })}
                      >
                        <SelectTrigger className="mt-1">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="idle">Idle</SelectItem>
                          <SelectItem value="running">Running</SelectItem>
                          <SelectItem value="completed">Completed</SelectItem>
                          <SelectItem value="error">Error</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    {/* Node-specific parameters */}
                    <Separator />
                    <div>
                      <Label className="text-sm font-medium">Parameters</Label>
                      <div className="mt-2 space-y-3">
                        {selectedNode.type === 'technical-indicators' && (
                          <>
                            <div>
                              <Label htmlFor="period" className="text-xs">Period</Label>
                              <Input
                                id="period"
                                type="number"
                                value={selectedNode.data.period || 20}
                                onChange={(e) => handleParameterChange('period', Number(e.target.value))}
                                min="1"
                                max="200"
                                className="mt-1"
                              />
                            </div>
                            <div>
                              <Label htmlFor="indicator" className="text-xs">Indicator Type</Label>
                              <Select
                                value={selectedNode.data.indicator || 'sma'}
                                onValueChange={(value) => handleParameterChange('indicator', value)}
                              >
                                <SelectTrigger className="mt-1">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="sma">Simple Moving Average</SelectItem>
                                  <SelectItem value="ema">Exponential Moving Average</SelectItem>
                                  <SelectItem value="rsi">RSI</SelectItem>
                                  <SelectItem value="macd">MACD</SelectItem>
                                  <SelectItem value="bollinger">Bollinger Bands</SelectItem>
                                </SelectContent>
                              </Select>
                            </div>
                          </>
                        )}

                        {selectedNode.type === 'risk-calculator' && (
                          <>
                            <div>
                              <Label htmlFor="risk-percent" className="text-xs">Risk Percentage</Label>
                              <div className="mt-2">
                                <Slider
                                  value={[selectedNode.data.riskPercentage || 2]}
                                  onValueChange={([value]) => handleParameterChange('riskPercentage', value)}
                                  min={0.1}
                                  max={10}
                                  step={0.1}
                                  className="w-full"
                                />
                                <div className="text-xs text-muted-foreground mt-1">
                                  {(selectedNode.data.riskPercentage || 2).toFixed(1)}%
                                </div>
                              </div>
                            </div>
                            <div className="flex items-center space-x-2">
                              <Switch
                                id="use-kelly"
                                checked={selectedNode.data.useKellyCriterion || false}
                                onCheckedChange={(checked) => handleParameterChange('useKellyCriterion', checked)}
                              />
                              <Label htmlFor="use-kelly" className="text-xs">Use Kelly Criterion</Label>
                            </div>
                          </>
                        )}

                        {selectedNode.type === 'signal-generator' && (
                          <>
                            <div>
                              <Label htmlFor="threshold" className="text-xs">Signal Threshold</Label>
                              <div className="mt-2">
                                <Slider
                                  value={[selectedNode.data.threshold || 0.5]}
                                  onValueChange={([value]) => handleParameterChange('threshold', value)}
                                  min={0}
                                  max={1}
                                  step={0.01}
                                  className="w-full"
                                />
                                <div className="text-xs text-muted-foreground mt-1">
                                  {(selectedNode.data.threshold || 0.5).toFixed(2)}
                                </div>
                              </div>
                            </div>
                            <div>
                              <Label htmlFor="signal-type" className="text-xs">Signal Type</Label>
                              <Select
                                value={selectedNode.data.signalType || 'buy_sell'}
                                onValueChange={(value) => handleParameterChange('signalType', value)}
                              >
                                <SelectTrigger className="mt-1">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="buy_sell">Buy/Sell</SelectItem>
                                  <SelectItem value="long_short">Long/Short</SelectItem>
                                  <SelectItem value="score">Confidence Score</SelectItem>
                                </SelectContent>
                              </Select>
                            </div>
                          </>
                        )}
                      </div>
                    </div>

                    <Separator />
                    <div className="flex gap-2">
                      <Button size="sm" className="flex-1">Apply</Button>
                      <Button variant="outline" size="sm" className="flex-1">Reset</Button>
                    </div>
                  </CardContent>
                </Card>
              </div>
            ) : (
              <div className="flex items-center justify-center h-32">
                <div className="text-center text-muted-foreground">
                  <Settings className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">Select a node to configure properties</p>
                </div>
              </div>
            )}
          </TabsContent>

          <TabsContent value="logs" className="p-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <Bug className="h-4 w-4" />
                  Node Logs
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="bg-black/20 rounded-lg p-3 font-mono text-xs h-64 overflow-auto">
                  {selectedNode ? (
                    <div className="space-y-1">
                      <div className="text-green-400">[INFO] Node {selectedNode.id} initialized</div>
                      <div className="text-blue-400">[DEBUG] Processing input data</div>
                      <div className="text-yellow-400">[WARN] High memory usage detected</div>
                      <div className="text-green-400">[INFO] Output generated successfully</div>
                    </div>
                  ) : (
                    <div className="text-muted-foreground">No node selected</div>
                  )}
                </div>
                <div className="flex gap-2 mt-3">
                  <Button variant="outline" size="sm">Clear</Button>
                  <Button variant="outline" size="sm">Export</Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="system" className="p-4 space-y-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <Activity className="h-4 w-4" />
                  System Metrics
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="flex items-center gap-2">
                      <Gauge className="h-4 w-4" />
                      CPU Usage
                    </span>
                    <span>{systemMetrics.cpuUsage}%</span>
                  </div>
                  <Progress value={systemMetrics.cpuUsage} className="mt-2 h-2" />
                </div>

                <div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="flex items-center gap-2">
                      <HardDrive className="h-4 w-4" />
                      Memory
                    </span>
                    <span>{systemMetrics.memoryUsage} MB</span>
                  </div>
                  <Progress value={Math.min((systemMetrics.memoryUsage / 1000) * 100, 100)} className="mt-2 h-2" />
                </div>

                <div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="flex items-center gap-2">
                      <Wifi className="h-4 w-4" />
                      Latency
                    </span>
                    <span>{systemMetrics.latency}ms</span>
                  </div>
                  <Progress 
                    value={Math.max(100 - systemMetrics.latency * 10, 0)} 
                    className="mt-2 h-2" 
                  />
                </div>

                <div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="flex items-center gap-2">
                      <Eye className="h-4 w-4" />
                      Data Flow Rate
                    </span>
                    <span>{systemMetrics.dataFlowRate}/s</span>
                  </div>
                  <Progress value={Math.min((systemMetrics.dataFlowRate / 200) * 100, 100)} className="mt-2 h-2" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Performance Tips</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Alert>
                  <Activity className="h-4 w-4" />
                  <div>
                    <div className="font-medium text-sm">Optimization Suggestions</div>
                    <div className="text-xs mt-1">
                      • Reduce node complexity for better performance<br/>
                      • Use caching for frequently accessed data<br/>
                      • Consider batch processing for large datasets
                    </div>
                  </div>
                </Alert>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default NodePropertiesPanel;