package basispoints

import (
	"container/list"
	"encoding/json"
	"fmt"
	"sort"
	"sync"
	"time"
)

// CatalogCache holds immutable, versioned client declarations, not responses.
// Only callers with an account/key/trusted-session scope may opt into reuse.
// Explicit tools replace the catalog (including []); omitted tools inherit it.
// Current additional_tools declarations supersede inherited versions. Conflicts
// inside the current request still go through Prepare's strict validation.
const catalogCacheIdleTTL = 30 * time.Minute

type CatalogCache struct {
	mu      sync.Mutex
	entries map[string]*list.Element
	order   list.List
	bytes   int
	version uint64
	now     func() time.Time
}
type catalogEntry struct {
	scope   string
	raw     []byte
	version uint64
	touched time.Time
}

func (c *CatalogCache) clock() time.Time {
	if c.now != nil {
		return c.now()
	}
	return time.Now()
}
func (c *CatalogCache) remove(e *list.Element) {
	v, _ := e.Value.(catalogEntry)
	delete(c.entries, v.scope)
	c.bytes -= len(v.raw)
	c.order.Remove(e)
}
func (c *CatalogCache) prune(now time.Time) {
	for e := c.order.Front(); e != nil; e = c.order.Front() {
		entry, _ := e.Value.(catalogEntry)
		if now.Sub(entry.touched) < catalogCacheIdleTTL {
			break
		}
		c.remove(e)
	}
}
func (c *CatalogCache) snapshot(scope string) ([]byte, uint64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := c.clock()
	c.prune(now)
	e := c.entries[scope]
	if e == nil {
		return nil, 0
	}
	v, _ := e.Value.(catalogEntry)
	v.touched = now
	e.Value = v
	c.order.MoveToBack(e)
	return v.raw, v.version
}
func (c *CatalogCache) commit(scope string, expected uint64, raw []byte) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := c.clock()
	c.prune(now)
	current := uint64(0)
	old := c.entries[scope]
	if old != nil {
		entry, _ := old.Value.(catalogEntry)
		current = entry.version
	}
	if current != expected {
		return false
	}
	if c.entries == nil {
		c.entries = make(map[string]*list.Element)
	}
	if old != nil {
		c.remove(old)
	}
	c.version++
	c.entries[scope] = c.order.PushBack(catalogEntry{scope: scope, raw: raw, version: c.version, touched: now})
	c.bytes += len(raw)
	for len(c.entries) > 512 || c.bytes > 16<<20 {
		c.remove(c.order.Front())
	}
	return true
}

func PrepareWithCatalog(raw []byte, scope string, replay *ReplayCache, cache *CatalogCache, catalogScopes ...string) ([]byte, *Bridge, error) {
	catalogScope := scope
	if len(catalogScopes) == 1 {
		catalogScope = catalogScopes[0]
	}
	return prepareWithCatalog(raw, scope, catalogScope, replay, cache)
}

func PrepareWithCatalogOptions(raw []byte, scope string, replay *ReplayCache, cache *CatalogCache, catalogScope string, options PrepareOptions) ([]byte, *Bridge, error) {
	return prepareWithCatalog(raw, scope, catalogScope, replay, cache, options)
}

func prepareWithCatalog(raw []byte, scope, catalogScope string, replay *ReplayCache, cache *CatalogCache, overrides ...PrepareOptions) ([]byte, *Bridge, error) {
	options := PrepareOptions{}
	if len(overrides) > 0 {
		options = overrides[0]
	}
	if cache == nil || scope == "" || catalogScope == "" {
		return PrepareWithOptions(raw, scope, replay, options)
	}
	var source object
	if decode(raw, &source) != nil || source == nil {
		return nil, nil, fmt.Errorf("invalid Basispoints request JSON")
	}
	if text(source["tool_choice"]) == "none" {
		return PrepareWithOptions(raw, scope, replay, options)
	}
	_, explicit := source["tools"]
	for attempt := 0; attempt < 8; attempt++ {
		previous, version := cache.snapshot(catalogScope)
		var candidate object
		if err := decode(raw, &candidate); err != nil {
			return nil, nil, err
		}
		if !explicit && previous != nil {
			var inherited []any
			if err := decode(previous, &inherited); err != nil {
				return nil, nil, err
			}
			inherited, err := inheritedCatalogTools(inherited, candidate["input"])
			if err != nil {
				return nil, nil, err
			}
			candidate["tools"] = inherited
		}
		encoded, err := json.Marshal(candidate)
		if err != nil {
			return nil, nil, err
		}
		body, b, err := PrepareWithOptions(encoded, scope, replay, options)
		if err != nil {
			return nil, nil, err
		}
		saved, err := json.Marshal(catalogDeclarations(b.tools))
		if err != nil {
			return nil, nil, err
		}
		if len(saved) > 1<<20 {
			cache.commit(catalogScope, version, []byte("[]")) // Revoke older declarations.
			return body, b, nil                               // Oversized valid catalogs work without being cached.
		}
		if cache.commit(catalogScope, version, saved) {
			return body, b, nil
		}
		// A concurrent explicit replacement is authoritative. This request keeps
		// its own validated catalog, but cannot overwrite the newer snapshot.
		if explicit {
			return body, b, nil
		}
		// Concurrent incremental declarations merge against a fresh snapshot.
	}
	return nil, nil, fmt.Errorf("basispoints tool catalog changed concurrently; retry request")
}

// Cached descriptions can change when a client's runtime tools are refreshed.
// Only missing declarations may be inherited; the current caller remains the
// authority. Validate all current additions together before removing any cached
// entry, so ambiguous declarations in one request are never silently accepted.
func inheritedCatalogTools(inherited []any, input any) ([]any, error) {
	current := &Bridge{tools: make(map[string]tool)}
	items, _ := input.([]any)
	for _, raw := range items {
		item, _ := raw.(object)
		if text(item["type"]) == "additional_tools" {
			if _, err := current.collectTools(item["tools"], ""); err != nil {
				return nil, err
			}
		}
	}
	if len(current.tools) == 0 {
		return inherited, nil
	}
	cached := &Bridge{tools: make(map[string]tool)}
	if _, err := cached.collectTools(inherited, ""); err != nil {
		return nil, err
	}
	for key := range current.tools {
		delete(cached.tools, key)
	}
	return catalogDeclarations(cached.tools), nil
}

func catalogDeclarations(tools map[string]tool) []any {
	keys := make([]string, 0, len(tools))
	for key := range tools {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	declarations := make([]any, 0, len(keys))
	for _, key := range keys {
		info := tools[key]
		var declaration any = info.Catalog
		if info.Namespace != "" {
			declaration = object{"type": "namespace", "name": info.Namespace, "tools": []any{declaration}}
		}
		declarations = append(declarations, declaration)
	}
	return declarations
}
