clc; close all;
an = ann('random','nobias',2,2,1);

m = 1e-3;

x = linspace(-300*m,300*m,300);
unit = ones(size(x))*m; 
x150 = 150*unit;
x50 = 50*unit;
x0 = 0*unit;

y1 = an.gilbert_multiplier(x0,x);
y2 = an.gilbert_multiplier(x150,x);
y3 = an.gilbert_multiplier(-x150,x);
y4 = an.gilbert_multiplier(x50,x);
y5 = an.gilbert_multiplier(-x50,x);

% f = figure("name","gilbert_multiplier");
% hold on
% plot(x,y1,'-');
% plot(x,y2,'-');
% plot(x,y3,'-');
% plot(x,y4,'-');
% plot(x,y5,'-');

x = linspace(0,1,500);
unit = ones(size(x))*m; 
x150 = 150*unit;
x250 = 10050*unit;
x0 = 0*unit;

z1 = an.vga_multiplier(x,x0);
z2 = an.vga_multiplier(x,x150);
z3 = an.vga_multiplier(x,-x150);
z4 = an.vga_multiplier(x,x250);
z5 = an.vga_multiplier(x,-x250);


f = figure("name","vga_multiplier");
hold on
plot(x,z1,'-');
plot(x,z2,'-');
plot(x,z3,'-');
plot(x,z4,'-');
plot(x,z5,'-');



%xhes'in ve berilin devresinde k için bir fonksiyon uydurmaya çalış..
