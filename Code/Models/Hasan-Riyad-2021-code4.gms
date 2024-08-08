*set time series (days)
SET t time (days) /1*126/;

*create alias of set t
ALIAS(t,t1);

*include cusomter demand dataset. Change to required dataset as needed
$include Hasan-Riyad-2021-data4.dat
Display demand;

*assign values to hyperparameters
PARAMETER epsilon /0.42/;
PARAMETER C /16.684/;

*cutoff point. these points are the training data
PARAMETER co /100/;

*step of 1
Scalar step /1/;

*parameter used for regression function
PARAMETER predict(t);

*forecasting horizon i.e. how many days we will forecast ahead
SET fh forecasting horizon /1*13/;

*training data
SET train(t);
train(t) = yes$(ord(t) LE co);

*sets the points in the time series that will be forecasted. greater than cutoff (training data). less
*than cutoff + step (will update every point after the cutoff)
SET forecast(t);
forecast(t) = yes$(ord(t) GE co AND ord(t) LE (co+step));

*demand related attributes
SET z /z1*z14/;
PARAMETER a(t,z);
a(t,z)$(train(t) OR forecast(t)) = (demand(t)$(ord(t) LE ord(z)) + demand(t-ord(z))$(ord(t) GT ord(z))) ;

*define variables that will be calculated in the solving steps
POSITIVE VARIABLES lam1(t), lam2(t), xi1(t), xi2(t) ;
VARIABLES mx, mn, beta ;

*kernel function
PARAMETER kern(t1,t) ;
*polynomial kernel
kern(t1,t)$(train(t) OR forecast(t)) = sqr(SUM(z, a(t1,z)*a(t,z))+1) ;

*chi sqaure kernel
*kern(t1,t)$(train(t) OR forecast(t)) = SUM(z, (2*(a(t1,z)*a(t,z)))/(a(t1,z)+a(t,z))) ;

*linear kernel
*kern(t1,t)$(train(t) OR forecast(t)) = SUM(z, a(t1,z)*a(t,z)) ;

*step 1 problem
EQUATIONS D1o, D1c1, D1c2(t), D1c3(t);

D1o..                    mx =e= -0.5*(SUM(t$train(t), SUM(t1$train(t1),
                                 (lam1(t)-lam2(t))*(lam1(t1)-lam2(t1))*kern(t1,t))))
                                 - epsilon*SUM(t$train(t), lam1(t)+lam2(t))
                                 + SUM(t$train(t), demand(t)*(lam1(t)-lam2(t))) ;

D1c1..                   SUM(t$train(t), lam1(t) - lam2(t)) =e= 0 ;
D1c2(t)$train(t)..       lam1(t) =l= C ;
D1c3(t)$train(t)..       lam2(t) =l= C ;

Model D1 step 1 /D1o, D1c1, D1c2, D1c3/;
*solved via nonlinear programming
Solve D1 using nlp max mx;

*dual variables are fixed
PARAMETER altlam1(t);
altlam1(t)$train(t) = lam1.l(t);
PARAMETER altlam2(t);
altlam2(t)$train(t) = lam2.l(t);

*step 2 problem
EQUATIONS  P1o, P1c1(t), P1c2(t);

P1o..                 mn =E= SUM(t$train(t), xi1(t) + xi2(t));
P1c1(t)$train(t)..    demand(t) - (SUM(t1$train(t1), (altlam1(t1)-altlam2(t1))*kern(t1,t))+beta) =L= epsilon + xi1(t) ;
P1c2(t)$train(t)..    SUM(t1$train(t1), (altlam1(t)-altlam2(t))*kern(t1,t)) + beta - demand(t) =L= epsilon + xi2(t) ;

Model P1 step 1 /P1o, P1c1, P1c2/;
*solved via linear programming
Solve P1 using lp min mn;

*reports values of selected variables
display altlam1;
display altlam2;
display beta.l;

*step 3 problem (regression)
predict(t)$forecast(t) = SUM(t1$train(t1), (altlam1(t1) - altlam2(t1))*kern(t1,t)) + beta.l;

*updates parameters
demand(t)$(train(t)) = demand(t);
demand(t)$(forecast(t)) = predict(t);

*recursive loop
loop (fh,

*updates kernel function
a(t,z)$(ord(t) LE (card(train)+1+ord(fh))) = demand(t)$(ord(t) LE ord(z)) + demand(t-ord(z))$(ord(t) GT ord(z));
kern(t1,t)$((ord(t) LE (card(train)+1+ord(fh))) AND (ord(t1) LE (card(train)+1+ord(fh))))  =  sqr(SUM(z, a(t1,z)*a(t,z))+1);

*calculates the predicted point for each point after the cutoff, for the forecasting horizon
co=co+step;
forecast(t)=NO;
forecast(t)=YES$(ord(t) GT co AND ord(t) LE (co+step));
predict(t)=0;
predict(t)$forecast(t) = SUM(t1$train(t1), (altlam1(t1) - altlam2(t1))*kern(t1,t)) + beta.l;

*updates parameters
demand(t)$(ord(t)<=co) = demand(t);
demand(t)$(forecast(t)) = predict(t);

*displays the forecasted points
display demand;

);










