// ESPP_CLASS
#ifndef _FIXEDPAIRLISTADRESS_HPP
#define _FIXEDPAIRLISTADRESS_HPP

#include "log4espp.hpp"
#include "types.hpp"

//#include "Particle.hpp"
#include "FixedPairList.hpp"
#include "esutil/ESPPIterator.hpp"
#include <boost/unordered_map.hpp>
#include <boost/signals2.hpp>


namespace espresso {
	class FixedPairListAdress : public FixedPairList {
	  public:
		FixedPairListAdress(shared_ptr <storage::Storage> _storage);
		~FixedPairListAdress();

		/** Add the given particle pair to the list on this processor if the
		particle with the lower id belongs to this processor.  Note that
		this routine does not check whether the pair is inserted on
		another processor as well.
		\return whether the particle was inserted on this processor.
		*/
		bool add(longint pid1, longint pid2);
		void onParticlesChanged();

		static void registerPython();

	  private:
		using PairList::add;
		static LOG4ESPP_DECL_LOGGER(theLogger);
	};
}

#endif
