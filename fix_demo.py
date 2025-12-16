with open('demo.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix line 193 (index 192)
lines[192] = '    print(f"\\n💾 Rehearsal buffer updated:")\\n'

with open('demo.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Fixed!")
