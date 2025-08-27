/**
 * Component Palette - Draggable components for strategy building
 */

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Search, Database, Settings, TrendingUp, Brain, Zap } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ComponentPaletteProps {
  categories: any;
  onComponentDrag: (component: any) => void;
  searchable?: boolean;
}

const ComponentPalette: React.FC<ComponentPaletteProps> = ({
  categories,
  onComponentDrag,
  searchable = false
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const filteredCategories = Object.entries(categories).filter(([key, category]: [string, any]) =>
    !searchQuery || category.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    category.components.some((comp: any) => comp.name.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  return (
    <div className="h-full flex flex-col bg-gray-900 text-white">
      <div className="p-4 border-b border-gray-700">
        <h3 className="text-lg font-semibold mb-3">Component Palette</h3>
        {searchable && (
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search components..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 bg-gray-800 border-gray-600"
            />
          </div>
        )}
      </div>

      <div className="flex-1 overflow-auto p-2 space-y-2">
        {filteredCategories.map(([categoryKey, category]: [string, any]) => (
          <Card key={categoryKey} className="bg-gray-800 border-gray-700">
            <CardHeader 
              className="pb-2 cursor-pointer"
              onClick={() => setSelectedCategory(selectedCategory === categoryKey ? null : categoryKey)}
            >
              <CardTitle className="text-sm flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="h-4 w-4" style={{ color: category.color }}>
                    {React.createElement(category.icon, { className: "h-4 w-4" })}
                  </div>
                  {category.name}
                </div>
                <Badge variant="outline" className="text-xs">
                  {category.components.length}
                </Badge>
              </CardTitle>
            </CardHeader>
            
            {(!selectedCategory || selectedCategory === categoryKey) && (
              <CardContent className="pt-0">
                <div className="space-y-1">
                  {category.components.map((component: any) => (
                    <div
                      key={component.id}
                      className="flex items-center gap-2 p-2 rounded hover:bg-gray-700 cursor-grab active:cursor-grabbing transition-colors"
                      draggable
                      onDragStart={() => onComponentDrag(component)}
                      onClick={() => onComponentDrag(component)}
                    >
                      <span className="text-lg">{component.icon}</span>
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium truncate">{component.name}</div>
                        <div className="text-xs text-muted-foreground truncate">
                          {component.description}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            )}
          </Card>
        ))}

        {filteredCategories.length === 0 && (
          <div className="flex items-center justify-center h-32">
            <div className="text-center text-muted-foreground">
              <Search className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No components found</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ComponentPalette;