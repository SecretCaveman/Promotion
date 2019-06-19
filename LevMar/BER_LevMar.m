classdef BER_LevMar < regmethod
    %% Bending Energy Regularized Gauss-Newton Method
    
    properties
        % regularization parameter in k-th Gauss-Newton step is alpha0 * alpha_step^k
        alpha0 = 1;
        alpha_step = 2/3;
         
        max_GN_steps = 50;
        
        GN_TOL_grad = 1e-3;
        GN_TOL_stepsize = 1e-3;
        
        

    end
    
    methods
        function R = BER_LevMar(par)
            R = R@regmethod(par);
            R = set_parameters(R,par);
        end
        
        function [x_k, stat,F] = solve(R,x_start,y_obs,F)
            % x_start: initial guess
            % y_obs  : observed data
            % F      : forward operator (see README.txt)
            
            R.it_step = 0;
            x_k = x_start;
            x_prev = x_start;
            F.bd.ReferenceTheta = x_prev(1:F.bd.Npts);
            
            [y_k,F] = F.evaluate(x_start);
            
            [stat,F] = R.error_stat_output(F,[],x_k,y_obs,y_k,x_start);
            R.condPrint(1,', alpha=%1.3e\n',R.alpha0);
            stat.GN_steps = [];
            
            [mustExit, R.stoprule] = R.stoprule.stop(F,x_k,y_k,y_obs,R);
            
            while ~mustExit
                R.it_step = R.it_step+1;
                regpar = R.alpha0 * R.alpha_step^(R.it_step);
                
                GN_step = 0;
                mustExitSubLoop = false;
                while ~mustExitSubLoop
                    GN_step = GN_step + 1;
                    update = R.computeUpdate(F,-(y_obs-y_k),x_prev,regpar);
                    %x_prev = x_k;
                    [x_k,minimum_reached] = R.Update(x_k, update,y_obs, F,regpar);
                    %F.bd.ReferenceTheta = x_prev(1:F.bd.Npts);
                    [y_k,F] = F.evaluate(x_k);
                    [stat,F] = R.error_stat_output(F,stat,x_k,y_obs,y_k,x_start);
                    mustExitSubLoop = R.stopInnerLoop(minimum_reached,GN_step);
                end
                x_prev = x_k;
                F.bd.ReferenceTheta = x_prev(1:F.bd.Npts);
                R.condPrint(1,', alpha=%1.1e, CGstp = %i\n',regpar,GN_step);
                stat.GN_steps = [stat.GN_steps, GN_step];  
                [mustExit, R.stoprule] = R.stoprule.stop(F,x_k,y_k,y_obs,R);
            end
            
            [stop_ind,x_k] = R.stoprule.select_index(F);
            stat.chosen_rec = stop_ind;
            
            if ~isempty(find(R.plot_steps==-1))
               F = F.plot(x_k,x_start,y_k,y_obs,-stop_ind);
            end
        end
        
        %% ------------------------------------------------------------------------
        function [newx,minimum_reached] = Update(R,oldx,h,y_obs, F,regpar)
            ShrinkFactor = 1/2;
            ArmijoConstant = 1/100;
            maxbiter = 10;
            t=1;
            M = F.bd;
            Mnew = F.bd;
            Mnew.coeff = oldx + t*h;
            [succeeded, Mnew] = Mnew.ConstaintProjection;

%             F2 = F;
%             F2.bd = Mnew;
%             F2 = F2.getF(F2);
            [y_k,F] = F.evaluate(oldx);
            [datFid,DdatFid,D2datFid] = R.getForwardOp(F,-1*(y_obs-y_k));
            [y_kNew,F] = F.evaluate(oldx+t*h);
            m = 1/(2*pi) * F.Ydim;
            datFidnew = 1/2*((y_obs-y_kNew)'*F.applyGramY(y_obs-y_kNew));
            %[datFidnew,DdatFidnew,D2datFidnew] = getForwardOp(F,F.y_obs-y_kNew);
            Enew = datFidnew + regpar * Mnew.BendingEnergy;
            E = datFid + regpar * M.BendingEnergy;
            grad = -( DdatFid + regpar * M.DBendingEnergy');
            biter = 0;

            %% F.res = ((norm(grad)< 1e-3) || (norm(x(1:n+3))<1e-3));
            while ((Enew > E + ArmijoConstant * t * (-grad)' * h) || (~succeeded)) && (biter < maxbiter) 
                biter = biter + 1;
                t = ShrinkFactor * t;
                Mnew.coeff = oldx + t*h;
                %Mnew = Displace(M, t * deltatheta', t * deltal, t * deltab');
                [succeeded, Mnew] = Mnew.ConstaintProjection;
                [y_kNew,F] = F.evaluate(Mnew.coeff);
                datFidnew = 1/2*((y_obs-y_kNew)'*F.applyGramY(y_obs-y_kNew));
                %[datFidnew,DdatFidnew,D2datFidnew] = getForwardOp(F,F.y_obs-y_kNew);
                Enew = datFidnew + regpar * Mnew.BendingEnergy;
            end
            newx = Mnew.coeff;

            if (biter == maxbiter) 
                fprintf('Warning maxbiter reached!!\n');
            end
            if ~succeeded
                error('Could not find back onto the manifold!');
            end
            minimum_reached = succeeded && ((norm(grad)< R.GN_TOL_grad) || (norm(h)<R.GN_TOL_stepsize));
        end
        
        %% ------------------------------------------------------------------------
        function deltaX = computeUpdate(R,F,y,xref,regpar)
            M=F.bd;
            
            %F=F.getF(F);
            [datFid,DdatFid,D2datFid] = R.getForwardOp(F,y);
            n = size(M.coeff,1)-3;

            DCInv = M.DClosureConstraintPinv;
            DE = M.DBendingEnergy;
            DD = DE*DCInv;
            Constr = M.LambdaD2ClosureConstraint(DD');
            hessE = blkdiag(M.D2BendingEnergy,zeros(3)) - Constr;
%             hessE = blkdiag(D2BendingEnergy(M),speye(3));

            H = sparse(D2datFid + regpar * hessE);
            B = sparse(DClosureConstraint(M));

            m = size(B,1);
            A = [H  , B' ; B , sparse(zeros(m,m)) ];  
            grad = -( DdatFid + regpar * M.DBendingEnergy');
            b=[grad ; zeros(m,1)];
            x = A\b;
            deltaX = x(1:(n+3));            
        end
        
        %% -------------------------------------------------------------------------
        function [datFid,DdatFid,D2datFid] = getForwardOp(R,F,y)
            %[y,F]=F.evaluate(F.bd.coeff);
            %F.y=y;
            %yobs=F.yobs;
            n = size(F.bd.coeff,1)-3;
            E=eye(n+3);
            m=F.Ydim;

            DF=zeros(m,n+3);
            for j=1:n+3
                h=E(:,j);
                DF(:,j)=F.derivative(h);
            end
            %F.DF=DF;
            m = 1/(2*pi) * F.Ydim;
            %H=1/2*(y-yobs)'*(y-yobs)/m;
            %DH=((y-yobs)'*DF)'/m ;
%             datFid=1/2*(y'*y)/m;
%             DdatFid=(y'*DF)'/m ;
%             D2datFid=DF'*DF/m;
            datFid=1/2*(y'*F.applyGramY(y));
            DdatFid=(y'*F.applyGramY(DF))';
            D2datFid=DF'*F.applyGramY(DF);
        end
        
        %% -------------------------------------------------------------------------
        function mustExitSubLoop = stopInnerLoop(R,minimum_reached,GN_step)
            mustExitSubLoop = ~((GN_step < R.max_GN_steps) && (~minimum_reached));
        end
        
    end
    
end