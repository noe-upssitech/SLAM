function [N,T,M,Z,arrayH,arrayR,in_fov,F,B,CC,Hfull,mX0,PX0,Qw,Rv,X] = dataSimulation(plot_p);
%DATASIMULATION
%  Simulation of a random experiment: robot motion, measurement collection, etc.
%  Syntax: [N,T,M,Z,arrayH,arrayR,in_fov,F,B,CC,Hfull,mX0,PX0,Qw,Rv,X] = dataSimulation(plot_p);
%
%  Input:
%  . plot_p : if 1, plots an animation, otherwise does nothing
%
%  Outputs:
%  . N: number of time samples
%  . T: vector of time samples (1xN)
%  . M: number of landmarks
%  . Z: 2MxN (2D) array of the outcomes of the measurement random process
%       (NaN entries correspond to unperceived landmarks; time is along the 2nd dimension)
%  . arrayH: 2Mx(2+2M)xN (3D) array of observation matrices
%            (NaN entries correspond to unperceived landmarks; time is along the 3rd dimension)
%  . arrayR: 2Mx2MxN (3D) array of measurement noise covariance matrices
%            (NaN entries correspond to unperceived landmarks; time is along the 3rd dimension)
%  . in_fov: 2MxN (2D) array of landmark visibility indexes
%            (NaN indexes correspond to unperceived landmarks; time is along the 2nd dimension)
%  -> as this is simulation, the following variables are also output
%     . F, B, CC: matrices/vectors involved in the prior dynamics of the full state vector
%                 (this (2+2M)x1 vector being made up with absolute coordinates of robot+landmarks)
%     . Hfull: 2Mx(2+2M) observation matrix if all landmarks were in the sensor fov
%     . mX0: (2+2M)x1 expectation vector of the initial state vector (at time 0)
%     . PX0: (2+2M)x(2+2M) covariance matrix of the initial state vector (at time 0)
%     . Qw: (2+2M)x(2+2M) covariance matrix of the (stationary) dynamics noise
%     . Rv: 2Mx2M covariance matrix of the (stationary) measurement noise if all landmarks were in the sensor fov
%     . X: (2+2M)xN (2D) array of the outcomes of the hidden state random process
%          (time is along the 2nd dimension)
%
%  Hint
%  . To suppress all lines full of NaN of a given matrix X, call X(any(~isnan(X),2),:);
%  . To suppress all lines and columns full of NaN for a given matrix X,
%    call X(any(~isnan(X),2),any(~isnan(X),1));
%
%  (c) Toulouse III Paul Sabatier University - P. Danès

N = 50;
deltaT = 1;
T = [0:N-1]*deltaT;
w = pi/4;

M = 5; % Number of landmarks
mX0robot = [0;0];
mX0landmarks = [-2; +4;
		-4; +2;
		-4; -2;
		-2; -4;
		+4; -2];
PX0robot = zeros(2,2);
PX0landmarks = .2 * eye(2*M);
PX0 = blkdiag(PX0robot,PX0landmarks);
cholPX0 = blkdiag(zeros(2,2),chol(PX0landmarks));
Qwrobot = 0.2^2 * eye(2);
Qwlandmarks = (1e-10)^2 * eye(2*M);
Rv = 0.15^2 * eye(2*M);

mX0 = [mX0robot;mX0landmarks];
Qw = blkdiag(Qwrobot,Qwlandmarks);
Frobot = blkdiag([cos(w*deltaT) -sin(w*deltaT) ; sin(w*deltaT) cos(w*deltaT)]);
CC = [-1.5;0]; % Center of ideal circle
Brobot = (eye(2) - Frobot) * CC;
F = blkdiag(Frobot, kron(eye(M),eye(2)));
B = [Brobot; kron(zeros(M,1),zeros(2,1))];

Hfull = [kron(ones(M,1),-eye(2)) eye(2*M)];
arrayH = nan([size(Hfull) N]);
arrayR = nan([size(Rv) N]);
X = nan*ones(2+2*M,N);
Z = nan*ones(2*M,N); 
in_fov = nan(2*M,N);

% Noise processes and State vector at initial time 0
W = chol(Qw)'*randn(2+2*M,N);
V = chol(Rv)'*randn(2*M,N);
X(:,1) = mX0 + cholPX0'*randn(2+2*M,1);
Z(:,1) = nan(2*M,1);

% Time instants 1:(N-1);
for k=2:N                              
  X(:,k) = F*X(:,k-1) + B + W(:,k-1); % noisy state
  Z(:,k) = Hfull*X(:,k); % noise-free measurement if everything is visible
  mat_Z = reshape(Z(:,k),2,M);
  mat_OR = kron(ones(1,M),(X(1:2,k)-CC));
  mat_OR_orth = kron(ones(1,M),[-(X(2,k)-CC(2)); X(1,k)-CC(1)]);
  %
  %  figure(1); plot(X(1,k),X(2,k),'ob'); hold on;
  %  for m = 1:M, plot(X(2*m+1,k),X(2*m+2,k),'xr'); end
  %
  in_fov(:,k) = ...
    transpose(kron(...
      double(and(...
        (dot(mat_Z,mat_OR) >= 0),...
        (dot(mat_Z,mat_OR_orth) >= 0))),...
      [1 1]));
  in_fov(:,k) = ...
    in_fov(:,k) ...
    ./ in_fov(:,k);
  Z(:,k) = in_fov(:,k) ...
           .* (Z(:,k) + V(:,k));
  arrayH(:,:,k) = kron(ones(1,2+2*M),in_fov(:,k)) .* Hfull;
  arrayR(:,:,k) = (in_fov(:,k)*in_fov(:,k)') .* Rv;
  %
  %  in_fov(:,k)
  %  Z(:,k)
  %  H(:,:,k)
  %  %
  %  disp('=   =   =');
  %  pause
end

if plot_p==1
figure(1);
for k=1:N
    hold off
    plot(CC(1),CC(2),'*k');
    hold on
    grid on
    axis equal
    xlim([-5 5]);
    ylim([-5 5]);
    title('GROUND TRUTH (STATE SPACE)')
    %
    plot(X(1,k),X(2,k),'o','markeredgecolor','none','markerfacecolor','b');
    plot(X(1,max(k-15,1):k),X(2,max(k-15,1):k),'--b');
    % visib_domain=[X(1,k) 5 5 X(1,k); X(2,k) X(2,k) 5 5];
    visib_domain=[X(1,k) X(1,k)+10*(X(1,k)-CC(1)) X(1,k)-100*(X(2,k)-CC(2)) ;
		  X(2,k) X(2,k)+10*(X(2,k)-CC(2)) X(2,k)+100*(X(1,k)-CC(1))];
    patch(visib_domain(1,:),visib_domain(2,:),'c','edgecolor','none','facealpha',.1);
    for m = 1:M
      if ~isnan(in_fov(2*m-1,k))
        color='g';
        plot([-5 +5],[X(2,k) X(2,k)],'--','color',[0.5 0.5 0.5],'linewidth',0.5);
        plot([X(1,k) X(1,k)],[-5 +5],'--','color',[0.5 0.5 0.5],'linewidth',0.5);
        plot([X((2*m+1),k) X((2*m+1),k)],[X(2,k) X((2*m+2),k)],':m','linewidth',0.5);
        text(X((2*m+1),k),X(2,k)-.2,sprintf('%s%s%s%s','(u^1-u^R)_','{',int2str(m),'}')); 
        plot([X((2*m+1),k) X(1,k)],[X((2*m+2),k) X((2*m+2),k)],':m','linewidth',0.5);
        text(X(1,k)-.8,X((2*m+2),k),sprintf('%s%s%s%s','(v^1-v^R)_','{',int2str(m),'}')); 
      else
        color='r';
      end
      plot(X((2*m+1),k),X((2*m+2),k),'s','markeredgecolor','k','markerfacecolor',color);
      text(X((2*m+1),k)+.3,X((2*m+2),k),sprintf('Amer %s',int2str(m)));
    end
    drawnow;
    pause(deltaT)
end

end
