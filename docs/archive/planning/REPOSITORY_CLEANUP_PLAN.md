---
draft: true
---

# Repository Cleanup Plan

**Date**: June 9, 2025
**Status**: Critical cleanup needed - 5 hours of development created significant cruft

## 🚨 **Current Problems**

### **1. Massive Log Pollution**
- **400+ log directories** in `logs/` from rapid testing
- Each test creates new timestamped directories
- Log files consuming significant disk space
- No retention policy

### **2. Root Directory Chaos**
- **30+ test files** scattered in root directory
- Multiple redundant markdown files (5+ CLI fix summaries)
- Temporary analysis files left behind
- Architecture diagrams misplaced

### **3. Session Files Explosion**
- **25+ session JSON files** in root `sessions/`
- No organization by date or purpose
- Mix of successful and failed sessions

### **4. Redundant Documentation**
- Multiple CLI fix summary files
- Overlapping user guides
- Archived docs mixed with current docs

## 🎯 **Cleanup Strategy**

### **Phase 1: Log Management (URGENT)**
```bash
# Keep only last 5 runs of each type, archive the rest
logs/
├── current/           # Last 5 successful runs only
├── archive/          # Older runs compressed
└── README.md         # Log retention policy
```

### **Phase 2: Root Directory Cleanup**
```bash
# Move all test files to tests/
# Consolidate documentation
# Organize analysis results
```

### **Phase 3: Session Management**
```bash
sessions/
├── current/          # Recent sessions
├── archive/          # Older sessions
└── README.md         # Session management guide
```

### **Phase 4: Documentation Consolidation**
```bash
docs/
├── README.md         # Main documentation index
├── user/            # User-facing docs only
├── development/     # Developer docs only
└── archive/         # Historical documents
```

## 📋 **Specific Actions**

### **Immediate (High Priority)**
1. **Archive old logs** - Keep only last 5 runs per type
2. **Move test files** - Root → tests/manual/ or delete obsolete ones
3. **Consolidate CLI docs** - Merge 5 CLI fix files into one
4. **Clean sessions** - Archive old, organize recent

### **Medium Priority**
5. **Update README** - Reflect current state after cleanup
6. **Standardize naming** - Consistent file/directory conventions
7. **Remove duplicates** - Identify and merge redundant files

### **Low Priority**
8. **Organize examples** - Better categorization
9. **Archive legacy** - Move old implementations to archive
10. **Documentation review** - Update outdated information

## 🎯 **Success Criteria**

- **Root directory**: <15 files (vs current 50+)
- **Logs**: <20 directories (vs current 400+)
- **Test files**: Organized in tests/ structure
- **Documentation**: Single source of truth for each topic
- **Sessions**: Organized by date/purpose

## ⚠️ **Risks & Mitigation**

- **Data Loss**: Create backup before deletion
- **Breaking Changes**: Test after major moves
- **Documentation Gaps**: Verify all links work after reorganization

This cleanup is essential for maintainability and will make the repository professional and navigable.
