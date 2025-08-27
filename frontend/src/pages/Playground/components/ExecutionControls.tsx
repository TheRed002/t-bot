/**
 * Execution Controls Component for Playground
 * Interface for controlling bot execution and monitoring settings
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

import {
  Play,
  Pause,
  Square,
  RotateCcw,
  Gauge,
  AlertTriangle,
  Info,
  Settings,
  Calendar,
  Building
} from 'lucide-react';

interface ExecutionControlsProps {
  isExecuting: boolean;
  executionStatus: string;
  onStart: () => void;
  onPause: () => void;
  onStop: () => void;
  onReset: () => void;
}

const ExecutionControls: React.FC<ExecutionControlsProps> = ({
  isExecuting,
  executionStatus,
  onStart,
  onPause,
  onStop,
  onReset
}) => {
  const [executionSpeed, setExecutionSpeed] = useState(1);
  const [realTimeMode, setRealTimeMode] = useState(false);
  const [startDate, setStartDate] = useState('2024-01-01');
  const [endDate, setEndDate] = useState('2024-12-31');
  const [initialCapital, setInitialCapital] = useState(10000);

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Play className="h-5 w-5" />
          Execution Controls
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Main Controls */}
        <div className="flex items-center gap-3">
          <Button
            onClick={onStart}
            disabled={isExecuting}
            className="gap-2"
          >
            <Play className="h-4 w-4" />
            Start
          </Button>
          <Button
            variant="outline"
            onClick={onPause}
            disabled={!isExecuting}
            className="gap-2"
          >
            <Pause className="h-4 w-4" />
            Pause
          </Button>
          <Button
            variant="outline"
            onClick={onStop}
            disabled={!isExecuting}
            className="gap-2"
          >
            <Square className="h-4 w-4" />
            Stop
          </Button>
          <Button
            variant="outline"
            onClick={onReset}
            className="gap-2"
          >
            <RotateCcw className="h-4 w-4" />
            Reset
          </Button>
        </div>

        {/* Status */}
        <div className="flex items-center gap-3">
          <Badge
            variant={isExecuting ? 'default' : 'secondary'}
            className="gap-1"
          >
            <div className={`w-2 h-2 rounded-full ${isExecuting ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`} />
            {executionStatus || 'Idle'}
          </Badge>
          {isExecuting && (
            <Progress value={Math.random() * 100} className="flex-1 h-2" />
          )}
        </div>

        {/* Execution Settings */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-4">
            <div>
              <Label htmlFor="execution-speed" className="flex items-center gap-2 mb-2">
                <Gauge className="h-4 w-4" />
                Execution Speed: {executionSpeed}x
              </Label>
              <Slider
                id="execution-speed"
                value={[executionSpeed]}
                onValueChange={([value]) => setExecutionSpeed(value)}
                min={0.1}
                max={10}
                step={0.1}
                className="w-full"
              />
            </div>

            <div className="flex items-center space-x-2">
              <Switch
                id="real-time"
                checked={realTimeMode}
                onCheckedChange={setRealTimeMode}
              />
              <Label htmlFor="real-time" className="flex items-center gap-2">
                <Info className="h-4 w-4" />
                Real-time mode
              </Label>
            </div>
          </div>

          <div className="space-y-4">
            <div>
              <Label htmlFor="initial-capital" className="flex items-center gap-2">
                <Building className="h-4 w-4" />
                Initial Capital
              </Label>
              <Input
                id="initial-capital"
                type="number"
                value={initialCapital}
                onChange={(e) => setInitialCapital(Number(e.target.value))}
                min="1000"
                step="1000"
                className="mt-1"
              />
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <Label htmlFor="start-date" className="text-sm">Start Date</Label>
                <Input
                  id="start-date"
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="mt-1"
                />
              </div>
              <div>
                <Label htmlFor="end-date" className="text-sm">End Date</Label>
                <Input
                  id="end-date"
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="mt-1"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Alerts */}
        {isExecuting && (
          <Alert>
            <Info className="h-4 w-4" />
            <div>
              <div className="font-medium">Execution in progress</div>
              <div className="text-sm">
                Monitor the performance metrics and be ready to stop if needed
              </div>
            </div>
          </Alert>
        )}

        {!isExecuting && executionStatus === 'error' && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <div>
              <div className="font-medium">Execution failed</div>
              <div className="text-sm">Check the logs for more information</div>
            </div>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default ExecutionControls;