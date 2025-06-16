### The Grand Workflow: From Zero to Running Experiments

**Assumptions:**
*   Your username is `user`.
*   Your home directory, and the only place you have write access, is `/home/misc/user/`.
*   The project files (`experiment.sub`, `workflow.dag`, `src/` folder) are available at a GitHub URL: `https://github.com/placeholder/deepzero-htcondor-example.git`.
*   Your large dataset is on your local computer.

---

### Phase 0: Initial Login and Environment Setup

**Goal:** Log into the submit node and set up a convenient environment variable for your unique home path.

1.  **Log into the HTCondor Submit Node:**
    ```bash
    ssh user@csl-htgc3.ad.cityu.edu.hk
    ```

2.  **Navigate to Your Designated Home and Verify:**
    ```bash
    cd /home/misc/user/
    pwd 
    # The output should be: /home/misc/user/
    ```

3.  **Set Up a Home Path Variable (Highly Recommended):**
    This makes the rest of the commands cleaner and easier to manage. This command sets a variable for your current session.
    ```bash
    export MY_HOME=/home/misc/user/
    echo "My working home is set to: $MY_HOME"
    ```

---

### Phase 1: One-Time Miniconda Installation

**Goal:** Install a personal Python environment manager since you cannot use system tools like `apt`.

1.  **Navigate to Your Home Directory:**
    ```bash
    cd $MY_HOME
    ```

2.  **Download the Miniconda Installer:**
    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    ```

3.  **Run the Installer:**
    ```bash
    bash Miniconda3-latest-Linux-x86_64.sh
    ```
    *   Press **Enter** to review the license.
    *   Type `yes` to accept the license.
    *   Press **Enter** to confirm the installation location (it will correctly default to `$MY_HOME/miniconda3`).
    *   Type `yes` to initialize Miniconda3.

4.  **Activate the New Shell Configuration:**
    This makes the `conda` command available to your current terminal session.
    ```bash
    source ~/.bashrc
    ```

5.  **Create and Activate Your Project's Conda Environment:**
    ```bash
    # Create an environment named 'deepzero-env' with Python 3.9
    conda create -n deepzero-env python=3.9 -y

    # Activate the environment
    conda activate deepzero-env
    ```

6.  **Install Required Python Packages:**
    ```bash
    # Install PyTorch for CUDA 11.8 (as an example)
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    
    # Install other common packages
    conda install pandas numpy -y
    ```
    You are now ready to set up your projects.

---

### Phase 2: One-Time Project and Data Setup

**Goal:** Get your code and data onto the cluster and prepare them for submission.

1.  **Create a Top-Level `projects` Directory:**
    ```bash
    cd $MY_HOME
    mkdir projects
    cd projects
    ```

2.  **Clone Your Project from GitHub:**
    This command downloads your project files into a folder named `deepzero_project`.
    ```bash
    git clone https://github.com/placeholder/deepzero-htcondor-example.git deepzero_project
    ```

3.  **Transfer Your Large Dataset to the Server:**
    **This command is run from your LOCAL COMPUTER's terminal, not from the SSH session.**
    ```bash
    # First, create the datasets directory on the server via your SSH session:
    # ssh user@csl-htgc3.ad.cityu.edu.hk "mkdir -p /home/misc/user/datasets"

    # Now, from your local machine, use scp (Secure Copy) to upload your data.
    # Replace the local path with the actual path to your dataset.
    scp -r /path/to/my_large_dataset/ user@csl-htgc3.ad.cityu.edu.hk:/home/misc/user/datasets/
    ```

4.  **Create the Compressed Dataset Archive:**
    **Switch back to your SSH session on `csl-htgc3`.** Navigate to your project directory.
    ```bash
    cd $MY_HOME/projects/deepzero_project/
    
    # Create the .tar.gz archive.
    # The -C flag is important: it changes directory before archiving, creating clean paths inside.
    echo "Creating compressed dataset archive..."
    tar -czf dataset.tar.gz -C $MY_HOME/datasets/ my_large_dataset
    echo "dataset.tar.gz created successfully."
    ```

5.  **Prepare the Project for Submission:**
    ```bash
    # Make sure the shell wrapper is executable
    chmod +x src/run_experiment.sh

    # Create the logs directory where output files will be saved
    mkdir logs
    ```

---

### Phase 3: Submitting and Monitoring Your Experiments (Recurring)

**Goal:** Run your experiments and check their progress. This is the phase you will repeat most often.

1.  **Navigate to Your Project Directory:**
    (You should already be here from the previous step, but it's good practice).
    ```bash
    cd $MY_HOME/projects/deepzero_project/
    ```

2.  **Submit the Entire Workflow:**
    This single command reads your `workflow.dag` file and submits all defined experiments to HTCondor.
    ```bash
    condor_submit_dag workflow.dag
    ```

3.  **Monitor the Job Queue:**
    ```bash
    # See the status of all your jobs (I=Idle, R=Running, C=Completed)
    condor_q user

    # To watch the queue update automatically every 5 seconds (press Ctrl+C to exit)
    watch -n 5 condor_q user
    ```

4.  **Check the Output of a Completed Job:**
    ```bash
    # List the contents of your logs directory
    ls -l logs/

    # View the standard output of a specific experiment
    cat logs/exp_1.out

    # View the error output (should be empty on success)
    cat logs/exp_1.err

    # To watch an output file in real-time as a job is running
    tail -f logs/exp_1.out
    ```

5.  **Manage Your Workflow:**
    ```bash
    # To remove a specific job (e.g., if you made a mistake)
    # Get the JOB_ID from `condor_q` (e.g., 12345.0)
    condor_rm 12345.0

    # To cancel ALL your jobs
    condor_rm user

    # If an outage occurs, find the rescue file and resume
    ls *.rescue.*
    condor_submit_dag workflow.dag.rescue.001
    ```
