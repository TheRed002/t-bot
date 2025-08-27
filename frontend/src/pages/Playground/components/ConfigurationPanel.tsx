/**
 * Configuration Panel Component for Playground
 */

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Settings } from 'lucide-react';

interface ConfigurationPanelProps {
  configuration: any;
  onConfigurationChange: (config: any) => void;
  isExecutionActive: boolean;
}

const ConfigurationPanel: React.FC<ConfigurationPanelProps> = (props) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings className="h-5 w-5" />
          Configuration Panel
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-center py-8">
          <Settings className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
          <p className="text-muted-foreground">Configuration panel coming soon</p>
        </div>
      </CardContent>
    </Card>
  );
};

export default ConfigurationPanel;