require "net/http"
require "openssl"

module Net
  class HTTP
    alias_method :orig_start, :start

    def start(*args, &block)
      if use_ssl?
        # Disable certificate verification for local development only.
        # This works around OpenSSL/CRL issues when fetching remote themes.
        self.verify_mode = OpenSSL::SSL::VERIFY_NONE
      end
      orig_start(*args, &block)
    end
  end
end


