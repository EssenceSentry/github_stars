New code repository!
As you might have noticed, the development of this repository has been limited to maintenance and bug fixes for some time now.  The reason is that there has been a collaborative effort to overhaul Spearmint into a new codebase.  This includes both algorithmic/theoretical and engineering improvements.  Check it out at https://github.com/HIPS/Spearmint.  Note that the new repository is under a non-commercial license with a contributor license agreement.  If you prefer not to agree to the license, you can freely use code here (though it is a bit older).

Spearmint
---------

Spearmint is a package to perform Bayesian optimization according to the
algorithms outlined in the paper:  

**Practical Bayesian Optimization of Machine Learning Algorithms**  
Jasper Snoek, Hugo Larochelle and Ryan P. Adams  
*Advances in Neural Information Processing Systems*, 2012  

This code is designed to automatically run experiments (thus the code
name 'spearmint') in a manner that iteratively adjusts a number of
parameters so as to minimize some objective in as few runs as
possible.

Spearmint is the result of a collaboration primarily between machine learning researchers at [Harvard University](https://hips.seas.harvard.edu/) and the [University of Toronto](http://learning.cs.toronto.edu/).

Dependencies
------------
This package requires:

* Python 2.7

* [Numpy](http://www.numpy.org/) version 1.6.1+
On Ubuntu linux you can install this package using the command:

		apt-get install python-numpy

* [Scipy](http://www.scipy.org/) version 0.9.0+
On Ubuntu linux you can install this package using the command:

		apt-get install python-scipy

* [Google Protocol Buffers](https://developers.google.com/protocol-buffers/) (for the fully automated code).
Note that you should be able to install protocol-buffers from source without requiring administrator privileges.  Otherwise, on Ubuntu linux you can install this package using the command:

		apt-get install python-protobuf
						
	and on Mac (if you use homebrew) with:

		brew install protobuf

	Then from within the spearmint sub-directory run the command:

		bin/make_protobufs

This package has been tested on Ubuntu linux (versions 11.0+) and
Mac-OSX.

The code consists of several parts.  It is designed to be modular to
allow swapping out various 'driver' and 'chooser' modules.  The
'chooser' modules are implementations of acquisition functions such as
expected improvement, UCB or random.  The drivers determine how
experiments are distributed and run on the system.  As the code is
designed to run experiments in parallel (spawning a new experiment as
soon a result comes in), this requires some engineering.  The current
implementations of these are in the 'spearmint' and 'spearmint-lite' subdirectories:

**Spearmint** is designed to automatically manage the launching and associated bookkeeping 
of experiments in either a single machine or cluster environment.  This requires that you provide
a 'wrapper' in a supported language (currently Python or Matlab) and a configuration file detailing parameters
to be tuned and their respective bounds. The wrapper must accept parameter values and then return simply a value which you wish to minimize with respect to the parameters. Spearmint will then iteratively call the wrapper with different 
parameter settings in an order that seeks to find the minimum value in as few evaluations (or cost) as possible.

**Spearmint-lite** is the 'bare-bones' stripped version of the code.
This version is simply driven by a flat-file and does not
automatically run experiments.  Instead, it proposes new experiments
(potentially multiple at a time) and requires that the user fill in
the result.  This is well suited to the case where writing a wrapper
around the code doesn't make sense (e.g. if the experiments don't
involve code at all) or if the user desires full control of the
process.  Also, the dependency on Google protocol buffers is replaced
with JSON.

Running the automated code: Spearmint
--------------------------------------------------------

The simplest way to get to know the code is probably to look at an
example.  In order to start a new experiment, you must create a directory 
that includes a wrapper script and config file.
We have created one simple example for you that optimizes the
'Braninhoo' benchmark in the subdirectory **examples/braninpy**.  Take a look at
**config.pb**.  It contains the specifications of the algorithm in
protocol buffer format.  In order to specify your optimization, you
have to fill in the variables 'language' (e.g. PYTHON or MATLAB) and
'name' (the name of the wrapper function you want to optimize).

Followed by these is a list of 'variables', which specifies the name,
type and size of the variables you wish to optimize over.  Each
variable must be either a FLOAT, INT or ENUM type, corresponding to
continuous real valued parameters, integer sequences and categorical
variables respectively.  MAX and MIN specify the bounds of the
variables over which to optimize and SIZE is the number of variables
of this type with these bounds.  Spearmint will call your wrapper
function with a dictionary type (in python) containing each of your
variables in a vector of size 'size', which you can access using the
name specified.  

Now take a look at branin.py (the wrapper which was
specified in the 'name' variable at the top of config.pb).  You will
see the file has a function 'main(job_id, params)'.  Your wrapper must
include this function, which spearmint will call passing in a job_id
(which is probably not interesting to you) and a dictionary, 'params',
containing the parameter vectors of the next experiment spearmint
wants to run.  The main function should take as input these parameters
and return a single real valued number representing the observed
function value (that is being optimized) at these inputs.

To install spearmint, go into the spearmint subdirectory and type (most likely preceded with 'sudo'):

	python setup.py install

You should add the spearmint subdirectory to your PYTHONPATH directory.  Note that you can often avoid the above step, but may have to add all the relevant modules to your PYTHONPATH and call spearmint directory from the directory using main.py.

To run spearmint, go into the **/bin** subdirectory and type:

	./spearmint ../examples/braninpy/config.pb --driver=local --method=GPEIOptChooser --method-args=noiseless=1

or alternatively in the **/spearmint** subdirectory:

	python main.py --driver=local --method=GPEIOptChooser --method-args=noiseless=1 ../examples/braninpy/config.pb

This will run spearmint according to the GP-EI MCMC strategy.  The code will sequentially spawn
processes that call the wrapper function and it will poll for results.
You will see that the code prints out the current best (i.e. lowest)
observation seen thus far and sequences of numbers corresponding to GP
hyperparameter samples and candidates it is optimizing over.  The
'method' argument specifies the chooser module (acquisition function)
to use and 'method-args' specifies chooser specific arguments.  In
this case, as braninhoo is an analytic function we tell the GP
hyperparameter sampling routine to not try to estimate noise.  This parameter is very important to specify correctly for the optimization to proceed properly.  If your algorithm is entirely deterministic (e.g. analytic) then specifying that it is noiseless will speed up the optimization considerably.  If your algorithm is not deterministic, as we expect for most machine learning algorithms and indeed most expensive experiments, then you should leave this out or set noiseless=0.

If you let it run for a while you will see that the current-best
decreases, eventually reaching the minimum at ~0.39. You can kill the
process (ctrl-c) at any time and you can restart from where it left off simply
by rerunning the spearmint command.

If you go back in to the braninpy directory you will see a number of
new files that spearmint uses to do bookkeeping.  Of particular
interest are **trace.csv** and the **output** directory.  **trace.csv**
contains a record of the experiments run so far and the best result
(and which experiment it came from) as a series over time.  Each line
of trace.csv contains the following in csv format: a timestamp, the
best value observed up to that timestamp, the job-id of the best value
observed, the number of potential candidates left, the number of
pending (currently running) experiments, the number of experiments
completed thus far.  The output directory contains a text file for
each job-id, containing the output of that job.  So if you want to
see, e.g. what the output (i.e. standard out and standard error) was
for the best job (as obtained from trace.csv) you can look up
job-id.txt in the output directory. 

 If you are debugging your code,
or the code is crashing for some reason, it's a good idea to look at
these files. Finally for ease of use, spearmint also prints out at
each iteration a file called 'best_job_and_result.txt' that contains the
best result observed so far, the job-id it came from and a dump of 
the names and values of all of the parameters corresponding to that result.

A script, bin/cleanup, is provided to completely restart an experiment
and delete all the results and intermediate files.  Simply run
`bin/cleanup <experiment_dir>`

Matlab code can also be optimized using this package. To do so, you
must specify in config.pb "type: Matlab" and use a matlab wrapper with
the "name" specified in config.pb.  The matlab wrapper must have a
function with the same name as the file name of the following form:
"function result = braninhoo(job_id, params)" Above we assume the file
name is "braninhoo.m".  Spearmint will pass in to this wrapper a
job_id and a matlab struct 'params' where the fields are given by the
variables specified in config.pb.  See the subdirectory "braninhoo"
for a matlab example matching that of python 'braninpy' described
above.

To run multiple jobs in parallel, pass to spearmint the argument:
`--max-concurrent=<#jobs>`

Spearmint is designed to be run in parallel either using multiple processors on a single machine or in a cluster environment.  These different environments, however, involve different queuing and fault-checking code and are thus coded as 'driver' modules.  Currently two drivers are available, but one can easily create a driver for a different environment by creating a new driver module (see the driver subdirectory for examples).

Using the `--driver=sge` flag, Spearmint can run on a system with Sun Grid Engine and it uses SGE to distribute experiments on a multi-node cluster in parallel using a queueing system in a fault-tolerant way.  It is particularly
well suited to the Amazon EC2 system.  Using [StarCluster](http://star.mit.edu/cluster/) will allow you to set up a large cluster and start distributing experiments within minutes.

Using the `--driver=local` flag will run Spearmint on a single machine with potentially many cores.  This driver simply spawns a new process on the current machine to run a new experiment.  This does not allow you to distribute across multiple machines, however.

### Web Status Page
Running spearmint with the flag `-w` will spawn a local web server that will display useful info and statistics about your optimization run in a much more interpretable and intuitive interface than the command line.  Spearmint will provide a link to the status page that you can simply paste into a browser window.


Running the basic code: Spearmint-lite 
---------------------------------------

Spearmint-lite is designed to be simple.  To run an experiment in
spearmint-lite, create a subdirectory as explained above.  Again, the
braninpy directory is provided as a demonstration.  In this case, the
experiment specification, which must be provided in config.json, is in
JSON format.  You must specify your problem as a sequence of JSON
objects.  As in the protocol buffer format above, each object must
have a name, a type (float, int or enum), a 'min', a 'max' and a
'size'. Nothing else needs to be specified.  

Go back to the top-level directory and run: 

	python spearmint-lite.py braninpy

Spearmint-lite will run one iteration of Bayesian
optimization and write out to a file named results.dat in the braninpy
subdirectory.  results.dat will contain a white-space delimited line
for each experiment, of the format: 
`<result> <time-taken> <list of parameters in the same order as config.json>`

Spearmint will propose new experiments and append them to results.dat each 
time it is run. Each proposed experiment will have a 'pending' result and 
time-taken, indicated by the letter P. The user must then run the experiment 
and fill in these values. Note that the time can safely be set to an arbitrary
value if the chooser module does not use it (only GPEIperSecChooser currently 
does). Spearmint will condition on the pending experiments when proposing new 
ones, so any number of experiments can be conducted in parallel.

A script, **cleanup.sh**, is provided to completely clean up all the intermediate
files and results in an experimental directory and restart the
experiment from scratch.

Chooser modules:
--------------- 

The chooser modules implement functions that tell spearmint which next
job to run.  Some correspond to 'acquisition functions' in the
Bayesian optimization literature.  Spearmint takes as an argument
`--method=ChooserModule` which allows one to easy swap out acquisition
functions. Choosers may optionally include parameters that can be
passed to spearmint using the argument
`--method-args=argument1,argument2,etc`.  These include, for example, the
number of GP hyperparameter samples to use. See the comments in
chooser files for chooser dependent arguments.  Below are described
the choosers provided in this package:

* **SequentialChooser:** Chooses the next experiment using a high
discrepancy Sobol sequence.  Experiments are taken sequentially from a
coarse-to-fine grid.

* **RandomChooser**: Experiments are sampled randomly from the unit hypercube.

* **GPEIOptChooser:** The GP EI MCMC algorithm from the paper. Jobs 
are first sampled densely from a dense grid on the unit hypercube
and then the best candidates are optimized 'fine-tuned' according
to EI.

* **GPEIperSecChooser:** The GP EI per Second algorithm from the paper.
Similar to GPEIOptChooser except points are optimized and evaluated
based on the EI per Second criterion, where each job is weighted
by the expected running time of the experiment.

* **GPEIChooser:** Points are densely sampled on the unit hypercube and the
best is returned according to the EI criterion.  This is considerably
faster than the GPEIOptChooser and works well in low-dimensional cases
where the grid covers the search space densely.

**IMPORTANT!**

When estimating noise, the Gaussian process prior over noise assumes
that the noise level is 'low' (e.g. between -1 to 1). If this is not
the case, make sure to rescale your function to make this true
(e.g. to approximately be between -1 and 1).  Otherwise, the algorithm
will find a bad high-noise mode that will result in bad performance.
