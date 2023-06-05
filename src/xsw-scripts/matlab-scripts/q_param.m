function [beta,gamma,delta,Eb] = q_param(datadir,Z,n,l,js,varargin)

if ~isempty(varargin)
    plotter = varargin{1};
else 
    plotter = 1;
end
% q_param(datadir,Z,n,l,js)
cd_old = cd;
cd(datadir)
fid = fopen('q_param.txt');
if isstring(l)~=1

    switch l
        case 0
            l='s';
        case 1

            l='p';
        case 2
            l='d';
        case 3
            l='f';
    end
   
end

switch l
    case 's'
        js = '';
    case 'p'
    	if js == 0.5
            js = '1/2';
        elseif js == 1.5
            js = '3/2';
        else
            error(['the p orbital cannot have a js of:' num2str(js)])
        end
    case 'd'
    	if js == 1.5
            js = '3/2';
        elseif js == 2.5
            js = '5/2';
        else
            error(['the d orbital cannot have a js of:' num2str(js)])            
        end
    case 'f'
    	if js == 2.5
            js = '5/2';
        elseif js == 3.5
            js = '7/2';
        else
            error(['the d orbital cannot have a js of:' num2str(js)])              
        end        
end
z={'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No'};

if plotter > 0
    disp(['##############' char(10) 'Measured orbital: ' z{Z} ' ' num2str(n) l js])
end 
name = [num2str(Z) ' ' num2str(n) l js];

line = [name(1:end-1) 'Z'];
nn = 0;
%line ~= name
while strcmp(line,name)~=1
    nn = nn +1;
    line = fgetl(fid);
    while length(line) < length(name)
        line = fgetl(fid);
        nn = nn+1;
    end
end
Eb = str2num(fgetl(fid));
Erow = str2num(fgetl(fid));

sigma = [Erow;str2num(fgetl(fid))];
beta = [Erow;str2num(fgetl(fid))];
gamma = [Erow;str2num(fgetl(fid))];
delta = [Erow;str2num(fgetl(fid))];

cd(cd_old)
