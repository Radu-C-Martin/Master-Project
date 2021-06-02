classdef MPCcasadi_v1_0 < matlab.System

    % Public, tunable properties
    properties(Nontunable)
        TimeStep = 0; % Time step MPC
        N        = 0; % Planning and control horizon N
        R        = 1; % Weights for control cost R 
        T        = 1; % Weights for slack variable for output constraints T 
        nState   = 0; % Number of states X
        nOut     = 0; % Number of outputs Y
        nIn      = 0; % Number of controlled inputs U
        nDst     = 0; % Number of disturbance inputs
        A        = 0; % A 
        Bd       = 0; % Bd (disturbance) 
        Bu       = 0; % Bu (control) 
        C        = 0; % C 
        D        = 0; % D 
        uMin     = 0; % Lower control constraints uMin
        uMax     = 0; % Upper control constraints uMax
        yMin     = 0; % Lower output constraints yMin
        yMax     = 0; % Upper output constraints yMax
    end

    properties(DiscreteState)

    end

    % Pre-computed constants
    properties(Access = private)
      casadi_solver
      lbg  
      ubg   
    end

    methods(Access = protected)
        function sts = getSampleTimeImpl(obj)
            sts = createSampleTime(obj, 'Type', 'Controllable', 'TickTime', obj.TimeStep); % Time step
        end
        
        function num = getNumInputsImpl(~) % Number of inputs
            num = 4;
        end
        function num = getNumOutputsImpl(~) % Number of outputs
            num = 5;
        end
        function [dt1, dt2, dt3, dt4, dt5] = getOutputDataTypeImpl(~) % Output data type
        	dt1 = 'double';
            dt2 = 'double';
            dt3 = 'double';
            dt4 = 'double';
            dt5 = 'double';
        end
        function dt1 = getInputDataTypeImpl(~) % Input data type
        	dt1 = 'double';
        end
        function [sz1, sz2, sz3, sz4, sz5] = getOutputSizeImpl(obj) % OUtput dimensions
        	sz1 = [1,        obj.nIn];     % mv
            sz2 = [obj.N+1,  obj.nState];  % xStar               
            sz3 = [obj.N,    obj.nOut];    % sStar
            sz4 = [obj.N,    obj.nIn];     % uStar
            sz5 = [1,        obj.nOut];    % yStarOut
        end
        function [sz1, sz2, sz3, sz4] = getInputSizeImpl(obj) % Input dimensions
        	sz1  = [obj.nState,  1];                 % xHat
            sz2  = [obj.N,       obj.nDst];          % disturbances
            sz3  = [obj.N,       1];                 % elec price
            sz4  = [1,           1];                 % on
        end
        function cp1 = isInputComplexImpl(~) % Inputs are complex numbers?
        	cp1 = false;
        end
        function [cp1, cp2, cp3, cp4, cp5] = isOutputComplexImpl(~) % Outputs are complex numbers?
        	cp1 = false;
            cp2 = false;
            cp3 = false;
            cp4 = false;            
            cp5 = false;
        end
        function fz1 = isInputFixedSizeImpl(~) % Input fixed size?
        	fz1 = true;
        end
        function [fz1, fz2, fz3, fz4, fz5] = isOutputFixedSizeImpl(~) % Output fixed size?
        	fz1 = true;
            fz2 = true;
            fz3 = true;
            fz4 = true;
            fz5 = true;
        end
        
        function setupImpl(obj)
            % Perform one-time calculations, such as computing constants
            import casadi.*

            %% Parameters
            nState  = obj.nState;
            nIn     = obj.nIn;
            nOut    = obj.nOut;
            nDst    = obj.nDst;
            N       = obj.N;
            R       = obj.R;
            T       = obj.T;
            A       = obj.A;
            Bd      = obj.Bd;
            Bu      = obj.Bu;
            C       = obj.C;
            D       = obj.D;

            %% Prepare variables
            U = MX.sym('U', nIn,     N); 
            P = MX.sym('P', nState + N + nDst*N); % Initial values, costElec, disturbances
            X = MX.sym('X', nState, (N+1));
            S = MX.sym('S', nOut,    N); % First state free

            J   = 0; % Objective function
            g   = [];  % constraints vector
            
            %% P indices
            iX0     = [1:nState];
            iCoEl   = [nState+1:nState+N];
            iDist   = [nState+N+1:nState+N+nDst*N];
            
            %% Disassemble P
            pX0     = P(iX0);
            pCoEl   = P(iCoEl);
            pDist   = reshape(P(iDist), [nDst N]); % Prone to shaping error

            %% Define variables
            states       = MX.sym('states',       nState);
            controls     = MX.sym('controls',     nIn);
            disturbances = MX.sym('disturbances', nDst);

            %% Dynamics
            f = Function('f',{P, states, controls, disturbances},{A*states + Bu*controls + Bd*disturbances});

            %% Compile all constraints
            g = [g; X(:,1) - pX0]; 

            for i = 1:N    
                g = [g; C*X(:,i+1) - S(:,i)];                           % State/output constraints, first state free
                g = [g; U(:,i)];                                        % Control constraints
                g = [g; X(:,i+1) - f(P, X(:,i), U(:,i), pDist(:,i))];   % System dynamics
                    
                % Cost function, first state given -> not punished
                J = J + R * U(:,i) * pCoEl(i) + S(:,i)'*T*S(:,i);
            end

            %% Reshape variables
            OPT_variables = veccat(X, S, U);

            %% Optimization
            nlp_mhe = struct('f', J,             ...
                             'x', OPT_variables, ...
                             'g', g,             ...
                             'p', P); 

            opts                   = struct;
            opts.ipopt.print_level = 0; %5;
            solver                 = nlpsol('solver', 'ipopt', nlp_mhe, opts);
            
            %% Pack opj
            obj.casadi_solver = solver;
            
        end

        function [mv, xStar, sStar, uStar, yStarOut] = stepImpl(obj, xHat, dist, cE, on) 
            % Implement algorithm. Calculate y as a function of input u and
            % discrete states.
            if on > 0.5
                %% Parameters
                nState  = obj.nState;
                N       = obj.N;
                nOut    = obj.nOut;
                nDst    = obj.nDst;
                nIn     = obj.nIn;
                yMin    = obj.yMin;
                yMax    = obj.yMax;
                uMin    = obj.uMin;
                uMax    = obj.uMax;
                C       = obj.C;
                solver  = obj.casadi_solver;
                Pdata   = [xHat; cE; reshape(dist', [nDst*N, 1])];   % Prone to shaping error!!!

                %% Constraints
                lbg = zeros(nState,1); % x0 constraints
                ubg = zeros(nState,1);
                
                % Output, control and dynamics constraints
                for i = 1:N
                    lbg = [lbg; yMin];
                    lbg = [lbg; uMin];
                    lbg = [lbg; zeros(nState,1)];

                    ubg = [ubg; yMax];
                    ubg = [ubg; uMax];
                    ubg = [ubg; zeros(nState,1)];
                end

                %% Solver
                sol    = solver('x0',  0,     ... % x0 = x* from before, shift one time step, double last time step
                                'lbg', lbg,   ...
                                'ubg', ubg,   ...
                                'p',   Pdata);

                %% Outputs
                xStar    = reshape(full(sol.x(1                    :nState*(N+1))),        [nState, (N+1)])';
                sStar    = reshape(full(sol.x(nState*(N+1)+1       :nState*(N+1)+nOut*N)), [nOut,    N])';
                uStar    = reshape(full(sol.x(nState*(N+1)+nOut*N+1:end)),                 [nIn,     N])';
                
                mv       =         full(sol.x(nState*(N+1)+nOut*N+1:nState*(N+1)+nOut*N+nIn))';
                yStarOut = C*xStar(2,:)'; % Second value is the target

            else % Zero output if MPC is disabled
                mv       = zeros(1,       obj.nIn);
                xStar    = zeros(obj.N+1, obj.nState);
                uStar    = zeros(obj.N,   obj.nIn);
                sStar    = zeros(obj.N,   obj.nOut);
                yStarOut = zeros(1,       obj.nOut);   
            end % \if on
        end % \stepImpl

        function resetImpl(obj)
            % Initialize / reset discrete-state properties
        end
    end
end
