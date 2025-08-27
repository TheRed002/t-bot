/**
 * Results Analysis Component for Playground
 */

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { BarChart3 } from 'lucide-react';

interface ResultsAnalysisProps {
  results: any;
  onExport: () => void;
  onCompare: () => void;
}

const ResultsAnalysis: React.FC<ResultsAnalysisProps> = (props) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5" />
          Results Analysis
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-center py-8">
          <BarChart3 className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
          <p className="text-muted-foreground">Results analysis coming soon</p>
        </div>
      </CardContent>
    </Card>
  );
};

export default ResultsAnalysis;