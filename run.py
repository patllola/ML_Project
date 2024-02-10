import subprocess

# Run the first Python script and capture its output
process = subprocess.Popen(["python3", "1dcom.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

# Check if the process was successful
if process.returncode == 0:
    # Save the output to a text file
    with open("/Users/sandeepreddy/Desktop/cloud/ML_Project/ML_Project/output.txt", "w") as output_file:
        output_file.write(stdout.decode())
        
    # with open("op.txt", "w") as op_file:
    #     subprocess.run(["python3", "pattern.py"], stdin=open("output.txt"), stdout=op_file)    
else:
    print("Error:", stderr.decode())
