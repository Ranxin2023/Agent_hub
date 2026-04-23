# CrewAI Installation Guide for Windows

Follow these steps exactly. Everything should be done in Command Prompt (not Python).

---

## Step 1: Open Command Prompt

1. Press the **Windows key** and **R** at the same time
2. Type: `cmd`
3. Press **Enter**
4. A black window will open - this is Command Prompt

---

## Step 2: Check if Python is Installed

1. In the Command Prompt window, type:
   ```
   python --version
   ```
2. Press **Enter**

**What you should see:**
- If Python is installed: `Python 3.10.x` or `Python 3.11.x` or `Python 3.12.x` or `Python 3.13.x`
- If Python is NOT installed: `'python' is not recognized as an internal or external command`

**If Python is NOT installed:**
1. Go to: https://www.python.org/downloads/
2. Click the big yellow button that says "Download Python"
3. Run the downloaded file
4. **IMPORTANT:** Check the box "Add Python to PATH" before clicking Install
5. Wait for installation to finish
6. Close and reopen Command Prompt
7. Go back to Step 2 and check again

---

## Step 3: Install CrewAI

1. Make sure you're in Command Prompt (the black window with `C:\>` at the start)
2. Type this exactly:
   ```
   pip install crewai
   ```
3. Press **Enter**
4. Wait for installation to finish (this may take 1-3 minutes)
5. You'll see "Successfully installed crewai" when it's done

---

## Step 4: Verify Installation

1. In the same Command Prompt window, type:
   ```
   python -c "import crewai; print('CrewAI installed successfully!')"
   ```
2. Press **Enter**

**What you should see:**
- `CrewAI installed successfully!`

If you see this message, you're done! CrewAI is installed correctly.

---

## Important Notes

- **Always use Command Prompt**, not Python. If you see `>>>` in your window, you're in Python. Type `exit()` and use Command Prompt instead.

- Command Prompt shows: `C:\>` at the start of each line
- Python shows: `>>>` at the start of each line

- All commands shown above must be typed in Command Prompt exactly as shown.

---

## Troubleshooting

**Problem:** `'pip' is not recognized`
- **Solution:** Use `python -m pip install crewai` instead of `pip install crewai`

**Problem:** Installation is very slow
- **Solution:** This is normal. Just wait for it to finish.

**Problem:** Error message about permissions
- **Solution:** Close Command Prompt, right-click on it, select "Run as administrator", then try again

---

## You're Done!

CrewAI is now installed on your Windows computer. You can start using it in your Python projects.

