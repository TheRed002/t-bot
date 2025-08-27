/**
 * Placeholder Component
 */

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Settings } from 'lucide-react';

interface PlaceholderProps {
  [key: string]: any;
}

const PlaceholderComponent: React.FC<PlaceholderProps> = (props) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings className="h-5 w-5" />
          Component Placeholder
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-center py-8">
          <Settings className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
          <p className="text-muted-foreground">Component coming soon</p>
        </div>
      </CardContent>
    </Card>
  );
};

export default PlaceholderComponent;
