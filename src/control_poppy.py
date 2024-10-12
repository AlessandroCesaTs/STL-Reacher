import pexpect

# Define your command and password
actions = [20, -89.59, 35.34, 0.44, 55.28, -4.84]
actions_str = ' '.join(map(str, actions))
command = f"ssh poppy@poppy.local 'source pyenv/bin/activate && python3 /home/poppy/move_robot.py {actions_str}'"
password = "poppy"

# Start the SSH process
child = pexpect.spawn(command)

# Look for the password prompt and send the password
child.expect("poppy@poppy.local's password: ")
child.sendline(password)

# Wait for the command to complete
child.interact()