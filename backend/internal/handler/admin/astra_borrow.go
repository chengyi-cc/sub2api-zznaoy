package admin

import (
	"errors"
	"net/http"
	"strconv"

	"github.com/Wei-Shaw/sub2api/internal/pkg/response"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
)

func (h *AccountHandler) astraBorrow(c *gin.Context) *service.AstraBorrowService {
	s := h.accountTestService.AstraBorrow()
	if s == nil {
		response.Error(c, http.StatusServiceUnavailable, "Astra borrowing unavailable")
	}
	return s
}

func (h *AccountHandler) GetAstraBorrow(c *gin.Context) {
	s := h.astraBorrow(c)
	if s == nil {
		return
	}
	v, err := s.Snapshot(c.Request.Context())
	if err != nil {
		response.Error(c, http.StatusServiceUnavailable, err.Error())
		return
	}
	response.Success(c, v)
}

func (h *AccountHandler) SaveAstraBorrow(c *gin.Context) {
	s := h.astraBorrow(c)
	if s == nil {
		return
	}
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 16384)
	var v service.AstraBorrowSettings
	if err := c.ShouldBindJSON(&v); err != nil {
		response.BadRequest(c, "Invalid Astra borrowing configuration")
		return
	}
	saved, err := s.Save(c.Request.Context(), v)
	if err != nil {
		if errors.Is(err, service.ErrAstraBorrowInvalid) {
			response.BadRequest(c, err.Error())
		} else {
			response.Error(c, http.StatusServiceUnavailable, "Astra configuration could not be saved")
		}
		return
	}
	response.Success(c, saved)
}

func (h *AccountHandler) VerifyAstraBorrow(c *gin.Context) {
	s := h.astraBorrow(c)
	if s == nil {
		return
	}
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil || id <= 0 {
		response.BadRequest(c, "Invalid target account")
		return
	}
	if err := s.Verify(c.Request.Context(), id); err != nil {
		if errors.Is(err, service.ErrAstraBorrowInvalid) {
			response.BadRequest(c, err.Error())
			return
		}
		if errors.Is(err, service.ErrAstraBorrowBusy) {
			response.Error(c, http.StatusConflict, err.Error())
			return
		}
		// A failed verification is a result, not an HTTP 200 business success.
		response.Error(c, http.StatusBadGateway, err.Error())
		return
	}
	response.Success(c, gin.H{"passed": true})
}

func (h *AccountHandler) AstraBorrowHistory(c *gin.Context) {
	s := h.astraBorrow(c)
	if s == nil {
		return
	}
	before := int64(0)
	if value := c.Query("before"); value != "" {
		parsed, err := strconv.ParseInt(value, 10, 64)
		if err != nil || parsed < 0 {
			response.BadRequest(c, "Invalid history cursor")
			return
		}
		before = parsed
	}
	items, err := s.History(c.Request.Context(), before)
	if err != nil {
		response.Error(c, http.StatusServiceUnavailable, "Astra history unavailable")
		return
	}
	response.Success(c, items)
}
